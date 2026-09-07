import json

import numpy as np
import pytest
import torch

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    BehaviorCloningRunner,
    MpsSafeAdaptiveAvgPool2d,
    compute_normalization,
    discover_episodes,
    extract_red_object_features,
    load_episodes,
    split_episodes,
    split_episodes_spatially,
)
from buddy_manipulator.dataset import save_episode
from buddy_manipulator.diffusion_policy import (
    DiffusionPolicy,
    DiffusionPolicyRunner,
    cosine_beta_schedule,
)
from buddy_manipulator.residual_diffusion import ResidualDiffusionRunner
from buddy_manipulator.evaluate_bc import evaluate_checkpoint
from buddy_manipulator.rollout_bc import load_policy_runner
from buddy_manipulator.train_bc import (
    MinimalReplacementSourceSampler,
    SourceBalancedSampler,
    failure_replay_sample_weights,
    train,
)
from buddy_manipulator.train_diffusion import train as train_diffusion
from buddy_manipulator.train_residual_diffusion import (
    train as train_residual_diffusion,
)


def write_episode(
    directory,
    index: int,
    *,
    success: bool,
    offset: float,
    source: str = "scripted",
    block_position: tuple[float, float, float] = (0.3, 0.08, 0.025),
) -> None:
    samples = 4
    rng = np.random.default_rng(index)
    arrays = {
        "timestamp": np.arange(samples, dtype=np.float64) * 0.2,
        "rgb": rng.integers(0, 256, (samples, 16, 16, 3), dtype=np.uint8),
        "depth": np.full((samples, 16, 16), 0.5 + offset, dtype=np.float32),
        "joint_position": np.tile(
            np.linspace(0.0, 0.5, 6, dtype=np.float32) + offset,
            (samples, 1),
        ),
        "action": np.tile(
            np.linspace(-0.2, 0.3, 6, dtype=np.float32) + offset,
            (samples, 1),
        ),
        "object_position": np.tile(
            np.asarray((0.3, 0.08, 0.025), dtype=np.float32),
            (samples, 1),
        ),
    }
    save_episode(
        directory,
        index,
        arrays,
        success=success,
        block_start_position=block_position,
        source=source,
    )


def test_episode_discovery_filters_failures_and_splits_by_episode(tmp_path) -> None:
    write_episode(tmp_path, 0, success=True, offset=0.0)
    write_episode(tmp_path, 1, success=False, offset=0.1)
    write_episode(tmp_path, 2, success=True, offset=0.2)

    episodes = discover_episodes(tmp_path)
    assert [episode.name for episode in episodes] == [
        "episode_00000",
        "episode_00002",
    ]
    assert [episode.source for episode in episodes] == ["scripted", "scripted"]
    assert episodes[0].block_position == (0.3, 0.08, 0.025)
    train_episodes, validation_episodes = split_episodes(
        episodes, validation_fraction=0.5, seed=3
    )
    assert {episode.name for episode in train_episodes}.isdisjoint(
        episode.name for episode in validation_episodes
    )


def test_spatial_split_preserves_each_populated_workspace_cell(tmp_path) -> None:
    episode_index = 0
    for x in (0.0, 1.0, 2.0):
        for y in (0.0, 1.0, 2.0):
            for repeat in range(3):
                write_episode(
                    tmp_path,
                    episode_index,
                    success=True,
                    offset=repeat * 0.01,
                    block_position=(x, y, 0.025),
                )
                episode_index += 1

    train_episodes, validation_episodes = split_episodes_spatially(
        discover_episodes(tmp_path),
        validation_fraction=1 / 3,
        seed=7,
        bins_per_axis=3,
    )

    assert len(train_episodes) == 18
    assert len(validation_episodes) == 9
    expected_cells = {(x, y) for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0, 2.0)}
    assert {
        episode.block_position[:2] for episode in train_episodes
    } == expected_cells
    assert {
        episode.block_position[:2] for episode in validation_episodes
    } == expected_cells


def test_spatial_split_retains_singleton_collection_source_in_training(tmp_path) -> None:
    for index in range(4):
        write_episode(tmp_path, index, success=True, offset=index * 0.01)
    write_episode(
        tmp_path,
        4,
        success=True,
        offset=0.04,
        source="teleop",
    )

    train_episodes, validation_episodes = split_episodes_spatially(
        discover_episodes(tmp_path),
        validation_fraction=0.4,
        seed=7,
    )

    assert len(validation_episodes) == 2
    assert any(episode.source == "teleop" for episode in train_episodes)


def test_dataset_and_policy_shapes(tmp_path) -> None:
    write_episode(tmp_path, 0, success=True, offset=0.0)
    episode_data = load_episodes(discover_episodes(tmp_path))
    normalization = compute_normalization(episode_data)
    dataset = BehaviorCloningDataset(episode_data, normalization)
    sample = dataset[0]

    assert sample["observation"].shape == (4, 16, 16)
    assert sample["joint_position"].shape == (6,)
    policy = BehaviorCloningPolicy()
    prediction = policy(
        sample["observation"].unsqueeze(0),
        sample["joint_position"].unsqueeze(0),
    )
    assert prediction.shape == (1, 6)

    chunk_dataset = BehaviorCloningDataset(
        episode_data,
        normalization,
        action_horizon=3,
    )
    assert chunk_dataset[0]["action"].shape == (3, 6)
    chunk_policy = BehaviorCloningPolicy(action_horizon=3)
    chunk_prediction = chunk_policy(
        sample["observation"].unsqueeze(0),
        sample["joint_position"].unsqueeze(0),
    )
    assert chunk_prediction.shape == (1, 3, 6)

    object_policy = BehaviorCloningPolicy(
        action_horizon=3,
        use_object_features=True,
    )
    object_prediction = object_policy(
        sample["observation"].unsqueeze(0),
        sample["joint_position"].unsqueeze(0),
    )
    assert object_prediction.shape == (1, 3, 6)


def test_red_object_features_preserve_image_location_and_depth() -> None:
    observation = torch.zeros((1, 4, 5, 5), dtype=torch.float32)
    observation[:, 0, 1, 3] = 1.0
    observation[:, 3, 1, 3] = 2.5

    features = extract_red_object_features(observation)

    assert features.shape == (1, 4)
    assert features[0].tolist() == pytest.approx((0.5, -0.5, 2.5, 4.0))
    assert extract_red_object_features(torch.zeros_like(observation))[0].tolist() == (
        pytest.approx((0.0, 0.0, 0.0, 0.0))
    )


def test_failure_replay_weights_target_requested_source_fraction(tmp_path) -> None:
    write_episode(
        tmp_path,
        0,
        success=True,
        offset=0.0,
        source="scripted",
    )
    write_episode(
        tmp_path,
        1,
        success=True,
        offset=0.1,
        source="failure_replay",
    )
    episodes = load_episodes(discover_episodes(tmp_path))
    normalization = compute_normalization(episodes)
    dataset = BehaviorCloningDataset(episodes, normalization)

    weights = failure_replay_sample_weights(dataset, 0.2).numpy()
    replay_weight = sum(
        weight
        for weight, source in zip(weights, dataset.sample_sources)
        if source == "failure_replay"
    )
    broad_weight = sum(
        weight
        for weight, source in zip(weights, dataset.sample_sources)
        if source != "failure_replay"
    )

    assert replay_weight == 0.2
    assert broad_weight == 0.8


def test_source_balanced_sampler_is_exact_unique_and_reproducible() -> None:
    sources = ["failure_replay"] * 8 + ["scripted"] * 8
    first = SourceBalancedSampler(sources, 0.25, seed=17)
    second = SourceBalancedSampler(sources, 0.25, seed=17)

    first_epoch = list(first)
    assert first_epoch == list(second)
    assert len(first_epoch) == len(set(first_epoch)) == 8
    assert sum(sources[index] == "failure_replay" for index in first_epoch) == 2
    assert sum(sources[index] != "failure_replay" for index in first_epoch) == 6

    assert list(first) == list(second)


def test_minimal_replacement_sampler_preserves_epoch_size_and_exact_ratio() -> None:
    sources = ["failure_replay"] * 8 + ["scripted"] * 8
    first = MinimalReplacementSourceSampler(sources, 0.25, seed=17)
    second = MinimalReplacementSourceSampler(sources, 0.25, seed=17)

    first_epoch = list(first)
    assert first_epoch == list(second)
    assert len(first_epoch) == 16
    assert len(set(first_epoch)) == 12
    assert sum(sources[index] == "failure_replay" for index in first_epoch) == 4
    assert sum(sources[index] != "failure_replay" for index in first_epoch) == 12

    assert list(first) == list(second)


def test_mps_safe_pool_matches_native_adaptive_pool() -> None:
    feature_map = torch.arange(2 * 3 * 15 * 20, dtype=torch.float32).reshape(
        2, 3, 15, 20
    )
    expected = torch.nn.functional.adaptive_avg_pool2d(feature_map, (2, 2))
    actual = MpsSafeAdaptiveAvgPool2d((2, 2))(feature_map)

    torch.testing.assert_close(actual, expected)


def test_training_writes_reusable_checkpoint(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    write_episode(dataset_dir, 0, success=True, offset=0.0)
    write_episode(dataset_dir, 1, success=True, offset=0.1)
    checkpoint_path, report = train(
        dataset_dir,
        output_dir,
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=3,
        split_seed=11,
        model_seed=13,
        sampler_seed=17,
        device_name="cpu",
        action_horizon=3,
    )

    assert checkpoint_path.exists()
    assert report["best_epoch"] == 1
    assert len(report["validation_episodes"]) == 1
    metrics = evaluate_checkpoint(
        checkpoint_path,
        dataset_dir,
        split="validation",
        batch_size=2,
        device_name="cpu",
    )
    assert metrics["samples"] == 4
    assert metrics["action_vectors"] == 12
    assert np.isfinite(metrics["normalized_mse"])
    saved_report = json.loads(
        (output_dir / "training_metrics.json").read_text(encoding="utf-8")
    )
    assert saved_report["best_epoch"] == 1
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["successful_only"] is True
    assert checkpoint["seed"] == 3
    assert checkpoint["split_seed"] == 11
    assert checkpoint["model_seed"] == 13
    assert checkpoint["sampler_seed"] == 17
    assert checkpoint["split_strategy"] == "random"
    assert checkpoint["source_sampling"] == "replacement"
    assert checkpoint["model_config"]["action_horizon"] == 3

    with np.load(dataset_dir / "episode_00000.npz") as arrays:
        runner = BehaviorCloningRunner.from_checkpoint(
            checkpoint_path,
            device_name="cpu",
        )
        action_chunk = runner.predict_chunk(
            arrays["rgb"][0],
            arrays["depth"][0],
            arrays["joint_position"][0],
        )
    assert action_chunk.shape == (3, 6)
    assert np.all(np.isfinite(action_chunk))


def test_source_sampling_strategy_requires_replay_fraction(tmp_path) -> None:
    write_episode(tmp_path, 0, success=True, offset=0.0)
    write_episode(tmp_path, 1, success=True, offset=0.1)

    with pytest.raises(ValueError, match="requires failure_replay_fraction"):
        train(
            tmp_path,
            tmp_path / "output",
            epochs=1,
            validation_fraction=0.5,
            source_sampling="without-replacement",
        )


def test_continual_training_preserves_split_and_adds_new_episode(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    for index in range(4):
        write_episode(dataset_dir, index, success=True, offset=index * 0.1)
    base_checkpoint, base_report = train(
        dataset_dir,
        tmp_path / "base",
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        seed=3,
        device_name="cpu",
        action_horizon=3,
        use_object_features=True,
    )
    write_episode(dataset_dir, 4, success=True, offset=0.4)

    checkpoint_path, report = train(
        dataset_dir,
        tmp_path / "continual",
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=5,
        device_name="cpu",
        action_horizon=3,
        use_object_features=True,
        initialize_from=base_checkpoint,
        preserve_checkpoint_split=True,
        reuse_checkpoint_normalization=True,
        freeze_image_encoder=True,
    )

    assert report["validation_episodes"] == base_report["validation_episodes"]
    assert set(base_report["train_episodes"]).issubset(report["train_episodes"])
    assert "episode_00004" in report["train_episodes"]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["preserve_checkpoint_split"] is True
    assert checkpoint["reuse_checkpoint_normalization"] is True
    assert checkpoint["freeze_image_encoder"] is True
    assert report["history"][0]["epoch"] == 0


def test_diffusion_policy_shapes_schedule_and_sampling() -> None:
    policy = DiffusionPolicy(
        action_horizon=3,
        diffusion_steps=8,
        condition_dim=16,
        hidden_dim=32,
        residual_blocks=1,
    )
    observation = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    joint_position = torch.zeros((2, 6), dtype=torch.float32)
    action = torch.zeros((2, 3, 6), dtype=torch.float32)
    noise = torch.ones_like(action)
    timesteps = torch.tensor((0, 7), dtype=torch.long)

    noisy_action = policy.add_noise(action, noise, timesteps)
    predicted_noise = policy(
        observation,
        joint_position,
        noisy_action,
        timesteps,
    )
    sampled_action = policy.sample(
        observation,
        joint_position,
        noise,
        inference_steps=4,
    )

    assert cosine_beta_schedule(8).shape == (8,)
    assert torch.all(policy.alphas_cumprod[1:] < policy.alphas_cumprod[:-1])
    assert predicted_noise.shape == action.shape
    assert sampled_action.shape == action.shape
    assert torch.all(torch.isfinite(sampled_action))


def test_diffusion_training_writes_rollout_compatible_checkpoint(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    write_episode(dataset_dir, 0, success=True, offset=0.0, source="scripted")
    write_episode(
        dataset_dir,
        1,
        success=True,
        offset=0.1,
        source="failure_replay",
    )
    checkpoint_path, report = train_diffusion(
        dataset_dir,
        output_dir,
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=3,
        action_horizon=3,
        diffusion_steps=4,
        inference_steps=2,
        condition_dim=16,
        hidden_dim=32,
        residual_blocks=1,
        failure_replay_fraction=None,
        device_name="cpu",
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["policy_type"] == "diffusion"
    assert checkpoint["model_config"]["action_horizon"] == 3
    assert report["status"] == "trained_pending_rollout"
    assert (output_dir / "experiment.json").exists()

    runner = load_policy_runner(checkpoint_path, device_name="cpu")
    assert isinstance(runner, DiffusionPolicyRunner)
    runner.configure_sampling(inference_steps=3, initial_noise_scale=0.0)
    assert runner.inference_steps == 3
    assert runner.initial_noise_scale == 0.0
    with np.load(dataset_dir / "episode_00000.npz") as arrays:
        runner.reset(11)
        first = runner.predict_chunk(
            arrays["rgb"][0],
            arrays["depth"][0],
            arrays["joint_position"][0],
        )
        runner.reset(11)
        second = runner.predict_chunk(
            arrays["rgb"][0],
            arrays["depth"][0],
            arrays["joint_position"][0],
        )
    assert first.shape == (3, 6)
    assert first == pytest.approx(second)


def test_residual_diffusion_checkpoint_preserves_bc_prior_interface(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    base_dir = tmp_path / "base"
    residual_dir = tmp_path / "residual"
    write_episode(dataset_dir, 0, success=True, offset=0.0)
    write_episode(dataset_dir, 1, success=True, offset=0.1)
    base_checkpoint, _ = train(
        dataset_dir,
        base_dir,
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=3,
        device_name="cpu",
        action_horizon=3,
    )
    checkpoint_path, report = train_residual_diffusion(
        dataset_dir,
        base_checkpoint,
        residual_dir,
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=3,
        action_horizon=3,
        diffusion_steps=4,
        inference_steps=2,
        failure_replay_fraction=None,
        device_name="cpu",
    )

    runner = load_policy_runner(checkpoint_path, device_name="cpu")
    assert isinstance(runner, ResidualDiffusionRunner)
    assert report["status"] == "trained_pending_rollout"
    runner.configure_sampling(residual_blend=0.0)
    with np.load(dataset_dir / "episode_00000.npz") as arrays:
        action = runner.predict_chunk(
            arrays["rgb"][0],
            arrays["depth"][0],
            arrays["joint_position"][0],
        )
    assert action.shape == (3, 6)
    assert np.all(np.isfinite(action))
