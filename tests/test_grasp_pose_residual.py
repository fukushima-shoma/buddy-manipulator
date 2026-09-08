from __future__ import annotations

import numpy as np
import pytest
import torch

from buddy_manipulator.dataset import save_episode
from buddy_manipulator.goal_task import ManipulationGoal
from buddy_manipulator.grasp_pose_residual import (
    GRASP_POSE_FEATURE_DIM,
    GraspPoseFeatureStats,
    GraspPoseResidualPolicy,
    GraspPoseResidualRunner,
    compute_feature_stats,
    grasp_pose_features,
)
from buddy_manipulator.sim_camera import RgbdFrame
from buddy_manipulator.train_grasp_pose_residual import train


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


def test_grasp_pose_features_include_selected_object_and_detection() -> None:
    features, detection = grasp_pose_features(
        red_block_frame(), ManipulationGoal("red", "green")
    )

    assert features.shape == (GRASP_POSE_FEATURE_DIM,)
    assert features[-2:].tolist() == [1.0, 0.0]
    assert detection.z == pytest.approx(0.0)


def test_feature_stats_are_safe_for_constant_dimensions() -> None:
    features = np.ones((3, GRASP_POSE_FEATURE_DIM), dtype=np.float32)

    stats = compute_feature_stats(features)

    assert np.all(stats.std >= 1e-6)
    with pytest.raises(ValueError, match="samples, 8"):
        compute_feature_stats(np.ones((3, 7), dtype=np.float32))


def test_residual_policy_starts_from_zero_correction() -> None:
    model = GraspPoseResidualPolicy()

    prediction = model(torch.ones((2, GRASP_POSE_FEATURE_DIM)))

    assert prediction.shape == (2, 3)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))


def test_residual_runner_clips_before_blending() -> None:
    model = GraspPoseResidualPolicy()
    with torch.no_grad():
        model.network[-1].bias.fill_(100.0)
    runner = GraspPoseResidualRunner(
        model=model,
        feature_stats=GraspPoseFeatureStats(
            mean=np.zeros(GRASP_POSE_FEATURE_DIM, dtype=np.float32),
            std=np.ones(GRASP_POSE_FEATURE_DIM, dtype=np.float32),
        ),
        device=torch.device("cpu"),
        blend=0.5,
    )
    _, baseline = grasp_pose_features(
        red_block_frame(), ManipulationGoal("red", "green")
    )

    corrected = runner.predict_detection(
        red_block_frame(), ManipulationGoal("red", "green")
    )

    assert np.asarray(corrected.position) - np.asarray(
        baseline.position
    ) == pytest.approx(np.full(3, 0.0075))


def write_goal_episode(directory, index: int, object_color: str, target_color: str) -> None:
    frame = red_block_frame()
    if object_color == "purple":
        frame.rgb[:, :, 2] = frame.rgb[:, :, 0]
    sample_count = 2
    goal = ManipulationGoal(object_color, target_color)
    arrays = {
        "timestamp": np.arange(sample_count, dtype=np.float64) * 0.2,
        "rgb": np.repeat(frame.rgb[None], sample_count, axis=0),
        "depth": np.repeat(frame.depth[None], sample_count, axis=0),
        "joint_position": np.zeros((sample_count, 6), dtype=np.float32),
        "action": np.zeros((sample_count, 6), dtype=np.float32),
        "object_position": np.repeat(
            np.asarray([[0.25, 0.0, 0.025]], dtype=np.float32),
            sample_count,
            axis=0,
        ),
        "goal": np.repeat(goal.vector()[None], sample_count, axis=0),
    }
    save_episode(
        directory,
        index,
        arrays,
        success=True,
        block_start_position=(0.25, 0.0, 0.025),
        source="goal_scripted",
        task_goal=goal.to_dict(),
    )


def test_training_writes_loadable_residual_checkpoint(tmp_path) -> None:
    write_goal_episode(tmp_path, 0, "red", "green")
    write_goal_episode(tmp_path, 1, "red", "yellow")
    write_goal_episode(tmp_path, 2, "purple", "yellow")

    checkpoint_path, report = train(
        tmp_path,
        tmp_path / "output",
        epochs=1,
        batch_size=1,
        validation_fraction=0.5,
        seed=7,
        device_name="cpu",
    )
    runner = GraspPoseResidualRunner.from_checkpoint(
        checkpoint_path, device_name="cpu"
    )

    assert checkpoint_path.exists()
    assert report["test_metrics"]["samples"] == 1
    assert runner.model.hidden_dim == 64
