from __future__ import annotations

import numpy as np
import pytest
import torch

from buddy_manipulator.goal_task import ManipulationGoal
from buddy_manipulator.grasp_pose_residual import grasp_pose_features
from buddy_manipulator.grasp_success_critic import (
    GRASP_CRITIC_INPUT_DIM,
    GraspCriticFeatureStats,
    GraspSuccessCritic,
    GraspSuccessCriticRunner,
    apply_grasp_residual,
    compute_critic_feature_stats,
    grasp_critic_features,
)
from buddy_manipulator.rollout_grasp_success_critic import candidate_residuals
from buddy_manipulator.sim_camera import RgbdFrame
from buddy_manipulator.train_grasp_success_critic import train


def red_block_frame() -> RgbdFrame:
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[8:13, 8:13, 0] = 255
    return RgbdFrame(
        rgb=rgb,
        depth=np.ones((20, 20), dtype=np.float32),
        vertical_fov_degrees=48.0,
        camera_position=np.asarray([0.0, 0.0, 1.0]),
        camera_rotation=np.eye(3),
    )


def test_critic_features_and_residual_application() -> None:
    pose = np.arange(8, dtype=np.float32)
    features = grasp_critic_features(pose, np.asarray([1.0, -2.0, 3.0]))
    _, detection = grasp_pose_features(
        red_block_frame(), ManipulationGoal("red", "green")
    )

    corrected = apply_grasp_residual(detection, np.asarray([1.0, -2.0, 3.0]))

    assert features.shape == (GRASP_CRITIC_INPUT_DIM,)
    expected = np.asarray(detection.position) + np.asarray([0.001, -0.002, 0.003])
    assert corrected.position == pytest.approx(expected)


def test_critic_stats_protect_constant_dimensions() -> None:
    stats = compute_critic_feature_stats(
        np.ones((3, GRASP_CRITIC_INPUT_DIM), dtype=np.float32)
    )
    assert np.all(stats.std >= 1e-6)


def test_runner_falls_back_when_score_gain_is_too_small() -> None:
    model = GraspSuccessCritic()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    runner = GraspSuccessCriticRunner(
        model=model,
        feature_stats=GraspCriticFeatureStats(
            mean=np.zeros(GRASP_CRITIC_INPUT_DIM, dtype=np.float32),
            std=np.ones(GRASP_CRITIC_INPUT_DIM, dtype=np.float32),
        ),
        device=torch.device("cpu"),
    )
    candidates = np.asarray([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)

    _, selected, probabilities = runner.choose_detection(
        red_block_frame(),
        ManipulationGoal("red", "green"),
        candidates,
        fallback_residual_mm=np.asarray([0.0, 2.0, 0.0]),
        minimum_score_improvement=0.01,
    )

    assert selected == pytest.approx([0.0, 2.0, 0.0])
    assert probabilities == pytest.approx([0.5, 0.5])


def test_candidate_grid_contains_biased_fallback_and_correction() -> None:
    candidates = candidate_residuals(6.0, step_mm=3.0, maximum_command_mm=9.0)
    assert np.any(np.all(np.isclose(candidates, [0.0, 6.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(candidates, [0.0, 0.0, 0.0]), axis=1))


def test_training_writes_loadable_checkpoint(tmp_path) -> None:
    scene_ids = np.repeat(np.arange(4), 2).astype(np.int32)
    residuals = np.tile(
        np.asarray([[0.0, 0.0, 0.0], [8.0, 8.0, 0.0]], dtype=np.float32),
        (4, 1),
    )
    features = np.zeros((8, GRASP_CRITIC_INPUT_DIM), dtype=np.float32)
    features[:, -3:] = residuals
    labels = np.tile(np.asarray([1.0, 0.0], dtype=np.float32), 4)
    dataset = tmp_path / "critic.npz"
    np.savez_compressed(
        dataset,
        features=features,
        success=labels,
        scene_id=scene_ids,
        residual_mm=residuals,
    )

    checkpoint, report = train(
        dataset,
        tmp_path / "output",
        epochs=1,
        batch_size=2,
        validation_fraction=0.25,
        seed=17,
        device_name="cpu",
    )
    runner = GraspSuccessCriticRunner.from_checkpoint(checkpoint, device_name="cpu")

    assert checkpoint.exists()
    assert report["validation_metrics"]["scenes"] == 1
    assert runner.model.hidden_dim == 64
