import json

import numpy as np
import torch

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    compute_normalization,
    discover_episodes,
    load_episodes,
    split_episodes,
)
from buddy_manipulator.dataset import save_episode
from buddy_manipulator.evaluate_bc import evaluate_checkpoint
from buddy_manipulator.train_bc import train


def write_episode(directory, index: int, *, success: bool, offset: float) -> None:
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
        device_name="cpu",
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
    assert np.isfinite(metrics["normalized_mse"])
    saved_report = json.loads(
        (output_dir / "training_metrics.json").read_text(encoding="utf-8")
    )
    assert saved_report["best_epoch"] == 1
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["successful_only"] is True
