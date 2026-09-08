from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    BehaviorCloningRunner,
    EpisodeData,
    MpsSafeAdaptiveAvgPool2d,
    compute_normalization,
    discover_episodes,
    extract_red_object_features,
    extract_goal_object_features,
    extract_goal_target_features,
    load_episodes,
    slice_goal_skill_episodes,
    split_episodes,
    split_episodes_spatially,
)
from buddy_manipulator.task_phase import GOAL_PHASE_DIM, encode_goal_phase
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
    task_goal: tuple[str, str] | None = None,
    samples: int = 4,
) -> None:
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
    goal_metadata = None
    if task_goal is not None:
        object_color, target_color = task_goal
        goal_vector = np.asarray(
            [
                object_color == "red",
                object_color == "purple",
                target_color == "green",
                target_color == "yellow",
            ],
            dtype=np.float32,
        )
        arrays["goal"] = np.repeat(goal_vector[None], samples, axis=0)
        goal_metadata = {
            "object_color": object_color,
            "target_color": target_color,
            "instruction": f"place the {object_color} block in the {target_color} zone",
        }
    save_episode(
        directory,
        index,
        arrays,
        success=success,
        block_start_position=block_position,
        source=source,
        task_goal=goal_metadata,
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

    goal_policy = BehaviorCloningPolicy(action_horizon=3, goal_dim=4)
    goal_prediction = goal_policy(
        sample["observation"].unsqueeze(0),
        sample["joint_position"].unsqueeze(0),
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    )
    assert goal_prediction.shape == (1, 3, 6)
    with pytest.raises(ValueError, match="goal must have shape"):
        goal_policy(
            sample["observation"].unsqueeze(0),
            sample["joint_position"].unsqueeze(0),
        )

    phase_dataset = BehaviorCloningDataset(
        episode_data,
        normalization,
        action_horizon=3,
        phase_conditioning=True,
    )
    assert phase_dataset[0]["phase"].shape == (GOAL_PHASE_DIM,)
    phase_policy = BehaviorCloningPolicy(
        action_horizon=3,
        phase_dim=GOAL_PHASE_DIM,
    )
    phase_prediction = phase_policy(
        sample["observation"].unsqueeze(0),
        sample["joint_position"].unsqueeze(0),
        phase=phase_dataset[0]["phase"].unsqueeze(0),
    )
    assert phase_prediction.shape == (1, 3, 6)
    with pytest.raises(ValueError, match="phase must have shape"):
        phase_policy(
            sample["observation"].unsqueeze(0),
            sample["joint_position"].unsqueeze(0),
        )


def test_goal_phase_encoding_follows_expert_boundaries() -> None:
    assert encode_goal_phase(0.0).argmax() == 0
    assert encode_goal_phase(1.19).argmax() == 0
    assert encode_goal_phase(1.2).argmax() == 1
    assert encode_goal_phase(13.4).argmax() == GOAL_PHASE_DIM - 1


def test_goal_skill_slices_overlap_around_grasp_handoff() -> None:
    sample_count = 68
    timestamp = np.arange(sample_count, dtype=np.float64) * 0.2
    episode = EpisodeData(
        name="episode_00000",
        source="goal_scripted",
        rgb=np.zeros((sample_count, 4, 4, 3), dtype=np.uint8),
        depth=np.zeros((sample_count, 4, 4), dtype=np.float32),
        joint_position=np.zeros((sample_count, 6), dtype=np.float32),
        action=np.zeros((sample_count, 6), dtype=np.float32),
        goal=np.ones((sample_count, 4), dtype=np.float32),
        timestamp=timestamp,
    )

    grasp = slice_goal_skill_episodes([episode], "grasp")[0]
    place = slice_goal_skill_episodes([episode], "place")[0]

    assert grasp.timestamp[-1] == pytest.approx(4.6)
    assert place.timestamp[0] == pytest.approx(3.2)
    assert np.all(grasp.goal[:, 2:4] == 0.0)
    assert np.all(place.goal[:, 0:2] == 0.0)
    assert np.all(place.goal[:, 2:4] == 1.0)
    assert grasp.sample_count + place.sample_count > episode.sample_count
    with pytest.raises(ValueError, match="unsupported goal skill"):
        slice_goal_skill_episodes([episode], "release")


def test_history_dataset_pads_episode_start_and_policy_uses_gru(tmp_path) -> None:
    write_episode(tmp_path, 0, success=True, offset=0.0)
    episode_data = load_episodes(discover_episodes(tmp_path))
    normalization = compute_normalization(episode_data)
    dataset = BehaviorCloningDataset(
        episode_data,
        normalization,
        action_horizon=3,
        history_horizon=3,
    )

    first_history = dataset[0]["joint_position"]
    assert first_history.shape == (3, 6)
    assert first_history[0].tolist() == pytest.approx(first_history[2].tolist())
    policy = BehaviorCloningPolicy(action_horizon=3, history_horizon=3)
    prediction = policy(
        dataset[0]["observation"].unsqueeze(0),
        first_history.unsqueeze(0),
    )
    assert prediction.shape == (1, 3, 6)
    with pytest.raises(ValueError, match="dimensions must be positive"):
        BehaviorCloningPolicy(history_horizon=0)


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


def test_goal_object_features_select_requested_color() -> None:
    observation = torch.zeros((2, 4, 5, 5), dtype=torch.float32)
    observation[:, 0, 1, 3] = 1.0
    observation[:, 3, 1, 3] = 2.5
    observation[:, 0, 3, 1] = 0.8
    observation[:, 2, 3, 1] = 1.0
    observation[:, 3, 3, 1] = 1.5
    goal = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    features = extract_goal_object_features(observation, goal)

    assert features[0].tolist() == pytest.approx((0.5, -0.5, 2.5, 4.0))
    assert features[1].tolist() == pytest.approx((-0.5, 0.5, 1.5, 4.0))


def test_goal_object_feature_policy_requires_goal_conditioning() -> None:
    with pytest.raises(ValueError, match="object-conditioned goal"):
        BehaviorCloningPolicy(use_goal_object_features=True)


def test_goal_target_features_select_requested_color() -> None:
    observation = torch.zeros((2, 4, 10, 10), dtype=torch.float32)
    observation[:, 1, 8, 5] = 0.9
    observation[:, 2, 8, 5] = 0.7
    observation[:, 3, 8, 5] = 2.0
    observation[:, 0, 7, 7] = 0.9
    observation[:, 1, 7, 7] = 0.8
    observation[:, 2, 7, 7] = 0.1
    observation[:, 3, 7, 7] = 1.5
    goal = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    features = extract_goal_target_features(observation, goal)

    assert features[0].tolist() == pytest.approx((1 / 9, 7 / 9, 2.0, 1.0))
    assert features[1].tolist() == pytest.approx((5 / 9, 5 / 9, 1.5, 1.0))


def test_goal_target_feature_policy_requires_goal_conditioning() -> None:
    with pytest.raises(ValueError, match="target-conditioned goal"):
        BehaviorCloningPolicy(use_goal_target_features=True)


def test_factorized_target_policy_selects_goal_specific_head() -> None:
    policy = BehaviorCloningPolicy(
        action_horizon=3,
        goal_dim=4,
        factorized_target_heads=True,
    )
    observation = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    joint_position = torch.zeros((2, 6), dtype=torch.float32)
    goals = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    prediction = policy(observation, joint_position, goals)

    assert prediction.shape == (2, 3, 6)
    assert len(policy.action_head) == 2
    with pytest.raises(ValueError, match="target-conditioned goal"):
        BehaviorCloningPolicy(factorized_target_heads=True)


def test_target_residual_policy_starts_from_shared_decoder() -> None:
    torch.manual_seed(3)
    shared = BehaviorCloningPolicy(action_horizon=3, goal_dim=4)
    torch.manual_seed(3)
    residual = BehaviorCloningPolicy(
        action_horizon=3,
        goal_dim=4,
        target_residual_heads=True,
    )
    observation = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    joint_position = torch.zeros((2, 6), dtype=torch.float32)
    goals = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        residual(observation, joint_position, goals),
        shared(observation, joint_position, goals),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        BehaviorCloningPolicy(
            goal_dim=4,
            factorized_target_heads=True,
            target_residual_heads=True,
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


def test_goal_training_keeps_combination_test_only(tmp_path) -> None:
    dataset_dir = tmp_path / "goals"
    combinations = [
        ("red", "green"),
        ("red", "yellow"),
        ("purple", "green"),
        ("purple", "yellow"),
    ]
    for index in range(8):
        write_episode(
            dataset_dir,
            index,
            success=True,
            offset=index * 0.01,
            task_goal=combinations[index % 4],
        )

    checkpoint_path, report = train(
        dataset_dir,
        tmp_path / "goal_output",
        epochs=1,
        batch_size=4,
        validation_fraction=0.2,
        seed=7,
        device_name="cpu",
        action_horizon=3,
        holdout_goal=("purple", "yellow"),
    )

    assert report["goal_dim"] == 4
    assert report["holdout_goal"] == ["purple", "yellow"]
    assert len(report["test_episodes"]) == 2
    assert report["test_metrics"]["samples"] == 8
    assert set(report["test_episodes"]).isdisjoint(report["train_episodes"])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_config"]["goal_dim"] == 4
    metrics = evaluate_checkpoint(
        checkpoint_path,
        dataset_dir,
        split="test",
        batch_size=4,
        device_name="cpu",
    )
    assert metrics["samples"] == 8


def test_goal_skill_training_slices_data_and_persists_contract(tmp_path) -> None:
    for index in range(2):
        write_episode(
            tmp_path,
            index,
            success=True,
            offset=index * 0.01,
            task_goal=("red", "green"),
            samples=30,
        )

    checkpoint_path, report = train(
        tmp_path,
        tmp_path / "place_output",
        epochs=1,
        batch_size=8,
        validation_fraction=0.5,
        seed=7,
        device_name="cpu",
        goal_skill="place",
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    runner = BehaviorCloningRunner.from_checkpoint(
        checkpoint_path, device_name="cpu"
    )
    metrics = evaluate_checkpoint(
        checkpoint_path,
        tmp_path,
        split="validation",
        batch_size=8,
        device_name="cpu",
    )
    assert report["goal_skill"] == checkpoint["goal_skill"] == "place"
    assert runner.goal_skill == "place"
    assert metrics["samples"] == 14


def test_phase_conditioned_training_writes_reusable_checkpoint(tmp_path) -> None:
    for index in range(2):
        write_episode(tmp_path, index, success=True, offset=index * 0.1)

    checkpoint_path, report = train(
        tmp_path,
        tmp_path / "phase_output",
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=5,
        device_name="cpu",
        action_horizon=3,
        phase_conditioning=True,
    )

    assert report["phase_conditioning"] is True
    assert report["phase_dim"] == GOAL_PHASE_DIM
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_config"]["phase_dim"] == GOAL_PHASE_DIM
    evaluate_checkpoint(
        checkpoint_path,
        tmp_path,
        split="validation",
        batch_size=2,
        device_name="cpu",
    )


def test_history_training_writes_stateful_runner_checkpoint(tmp_path) -> None:
    for index in range(2):
        write_episode(tmp_path, index, success=True, offset=index * 0.1)

    checkpoint_path, report = train(
        tmp_path,
        tmp_path / "history_output",
        epochs=1,
        batch_size=2,
        validation_fraction=0.5,
        seed=5,
        device_name="cpu",
        action_horizon=3,
        history_horizon=3,
    )

    assert report["history_horizon"] == 3
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["model_config"]["history_horizon"] == 3
    with np.load(tmp_path / "episode_00000.npz") as arrays:
        runner = BehaviorCloningRunner.from_checkpoint(
            checkpoint_path,
            device_name="cpu",
        )
        runner.predict_chunk(
            arrays["rgb"][0],
            arrays["depth"][0],
            arrays["joint_position"][0],
        )
        assert len(runner._joint_history) == 1
        runner.reset()
        assert runner._joint_history == []


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
