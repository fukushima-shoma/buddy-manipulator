from __future__ import annotations

import numpy as np
import pytest
import torch

from buddy_manipulator.goal_task import ManipulationGoal
from buddy_manipulator.placement_success_critic import (
    PLACEMENT_CRITIC_INPUT_DIM,
    PlacementFeatureStats,
    PlacementSuccessCritic,
    PlacementSuccessCriticRunner,
    compute_placement_feature_stats,
    grid_release_residuals,
    placement_critic_features,
)
from buddy_manipulator.train_placement_success_critic import train


def test_placement_features_and_candidate_grid() -> None:
    context = np.arange(9, dtype=np.float32)
    features = placement_critic_features(context, np.asarray([10.0, -20.0]))
    candidates = grid_release_residuals(
        np.asarray([0.0, 20.0]), step_mm=10.0, maximum_command_mm=30.0
    )

    assert features.shape == (PLACEMENT_CRITIC_INPUT_DIM,)
    assert np.any(np.all(np.isclose(candidates, [0.0, 20.0]), axis=1))
    assert np.any(np.all(np.isclose(candidates, [0.0, 0.0]), axis=1))


def test_placement_stats_protect_constant_dimensions() -> None:
    stats = compute_placement_feature_stats(
        np.ones((4, PLACEMENT_CRITIC_INPUT_DIM), dtype=np.float32)
    )
    assert np.all(stats.std >= 1e-6)


def test_placement_runner_falls_back_without_score_gain(monkeypatch) -> None:
    model = PlacementSuccessCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    runner = PlacementSuccessCriticRunner(
        model,
        PlacementFeatureStats(
            np.zeros(PLACEMENT_CRITIC_INPUT_DIM, dtype=np.float32),
            np.ones(PLACEMENT_CRITIC_INPUT_DIM, dtype=np.float32),
        ),
        torch.device("cpu"),
    )
    monkeypatch.setattr(
        "buddy_manipulator.placement_success_critic.placement_context_features",
        lambda *_args: np.zeros(9, dtype=np.float32),
    )
    candidates = np.asarray([[20.0, 0.0], [0.0, 10.0]], dtype=np.float32)

    selected, probabilities = runner.choose_residual(
        object(),
        ManipulationGoal("red", "green"),
        np.zeros(3),
        candidates,
        fallback_xy_mm=np.asarray([0.0, 10.0]),
        minimum_score_improvement=0.01,
    )

    assert selected == pytest.approx([0.0, 10.0])
    assert probabilities == pytest.approx([0.5, 0.5])


def test_placement_training_writes_loadable_checkpoint(tmp_path) -> None:
    scene_ids = np.repeat(np.arange(4), 2).astype(np.int32)
    residuals = np.tile(
        np.asarray([[0.0, 0.0], [70.0, 70.0]], dtype=np.float32), (4, 1)
    )
    features = np.zeros((8, PLACEMENT_CRITIC_INPUT_DIM), dtype=np.float32)
    features[:, -2:] = residuals
    dataset = tmp_path / "placements.npz"
    np.savez_compressed(
        dataset,
        features=features,
        success=np.tile(np.asarray([1.0, 0.0], dtype=np.float32), 4),
        scene_id=scene_ids,
        residual_xy_mm=residuals,
    )

    checkpoint, report = train(
        dataset,
        tmp_path / "output",
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        seed=23,
        device_name="cpu",
    )
    runner = PlacementSuccessCriticRunner.from_checkpoint(
        checkpoint, device_name="cpu"
    )

    assert report["validation_metrics"]["scenes"] == 1
    assert runner.model.hidden_dim == 64
