import json

import numpy as np
import torch

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    BehaviorCloningRunner,
    MpsSafeAdaptiveAvgPool2d,
    compute_normalization,
    discover_episodes,
    load_episodes,
    split_episodes,
)
from buddy_manipulator.dataset import save_episode
from buddy_manipulator.evaluate_bc import evaluate_checkpoint
from buddy_manipulator.train_bc import failure_replay_sample_weights, train


def write_episode(
    directory,
    index: int,
    *,
    success: bool,
    offset: float,
    source: str = "scripted",
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
        block_start_position=(0.3, 0.08, 0.025),
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
    train_episodes, validation_episodes = split_episodes(
        episodes, validation_fraction=0.5, seed=3
    )
    assert {episode.name for episode in train_episodes}.isdisjoint(
        episode.name for episode in validation_episodes
    )


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
