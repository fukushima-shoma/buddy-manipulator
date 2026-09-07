"""Vision-gated hybrid policy helpers for reliable workspace coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Optional

import numpy as np

from buddy_manipulator.kinematics import Pose, inverse_kinematics
from buddy_manipulator.vision import detect_red_object, locate_detection_in_world


WORKSPACE_X_BOUNDS_M = (0.27, 0.33)
WORKSPACE_Y_BOUNDS_M = (0.05, 0.11)
WORKSPACE_CAMERA_FOV_DEGREES = 48.0
WORKSPACE_CAMERA_POSITION_M = (0.25, 0.0, 0.75)
WORKSPACE_CAMERA_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

PositionEstimator = Callable[
    [np.ndarray, np.ndarray], Optional[tuple[float, float, float]]
]


def estimate_red_block_position(
    rgb: np.ndarray,
    depth: np.ndarray,
) -> tuple[float, float, float] | None:
    """Localize the red block using the calibrated fixed workspace camera."""
    pixel = detect_red_object(rgb)
    if pixel is None:
        return None
    detection = locate_detection_in_world(
        pixel,
        depth,
        vertical_fov_degrees=WORKSPACE_CAMERA_FOV_DEGREES,
        camera_position=np.asarray(WORKSPACE_CAMERA_POSITION_M),
        camera_rotation=np.asarray(WORKSPACE_CAMERA_ROTATION),
    )
    return detection.position


def is_workspace_edge(
    position: tuple[float, float, float],
    *,
    margin: float,
    x_bounds: tuple[float, float] = WORKSPACE_X_BOUNDS_M,
    y_bounds: tuple[float, float] = WORKSPACE_Y_BOUNDS_M,
) -> bool:
    """Return whether a position lies in the outer workspace band."""
    if margin <= 0:
        raise ValueError("edge margin must be positive")
    if margin * 2 >= min(x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0]):
        raise ValueError("edge margin leaves no central workspace")
    x, y, _ = position
    return not (
        x_bounds[0] + margin <= x <= x_bounds[1] - margin
        and y_bounds[0] + margin <= y <= y_bounds[1] - margin
    )


def _joint_action(pose: Pose, gripper: float) -> np.ndarray:
    joints = inverse_kinematics(pose, elbow_up=True)
    return np.asarray((*joints.as_tuple(), gripper, gripper), dtype=np.float64)


def build_scripted_grasp_actions(
    position: tuple[float, float, float],
) -> np.ndarray:
    """Build the same observable vision/IK grasp sequence used by the expert."""
    x, y, z = position
    pregrasp = _joint_action(Pose(x, y, z + 0.10, -math.pi / 2), 0.03)
    grasp_open = _joint_action(Pose(x, y, z, -math.pi / 2), 0.03)
    grasp_closed = _joint_action(Pose(x, y, z, -math.pi / 2), 0.0)
    lift = _joint_action(Pose(x, y, z + 0.15, -math.pi / 2), 0.0)
    return np.concatenate(
        [
            np.repeat(pregrasp[None], 8, axis=0),
            np.repeat(grasp_open[None], 6, axis=0),
            np.repeat(grasp_closed[None], 5, axis=0),
            np.repeat(lift[None], 8, axis=0),
        ]
    )


@dataclass
class ScriptedVisionGraspPolicyRunner:
    """Expose the deterministic RGB-D/IK expert through the policy interface."""

    action_horizon: int
    position_estimator: PositionEstimator = estimate_red_block_position
    target_position: tuple[float, float, float] | None = None
    _actions: np.ndarray | None = field(default=None, init=False, repr=False)
    _action_index: int = field(default=0, init=False, repr=False)
    device: str = field(default="cpu", init=False)

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action horizon must be positive")

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.target_position = None
        self._actions = None
        self._action_index = 0

    def set_target(self, position: tuple[float, float, float]) -> None:
        self.target_position = position
        self._actions = None
        self._action_index = 0

    def predict_chunk(self, rgb, depth, joint_position) -> np.ndarray:
        del joint_position
        if self._actions is None:
            position = self.target_position or self.position_estimator(rgb, depth)
            if position is None:
                raise RuntimeError("vision expert could not detect the red block")
            self.target_position = position
            self._actions = build_scripted_grasp_actions(position)
        if self._action_index >= len(self._actions):
            return np.repeat(self._actions[-1:], self.action_horizon, axis=0)
        end = self._action_index + self.action_horizon
        chunk = self._actions[self._action_index : end]
        self._action_index = end
        if len(chunk) < self.action_horizon:
            chunk = np.concatenate(
                [
                    chunk,
                    np.repeat(chunk[-1:], self.action_horizon - len(chunk), axis=0),
                ]
            )
        return chunk.copy()

    def predict(self, rgb, depth, joint_position) -> np.ndarray:
        return self.predict_chunk(rgb, depth, joint_position)[0]


@dataclass
class SpatialGatedPolicyRunner:
    """Route edge placements to a specialist and keep the learned center policy."""

    primary: Any
    specialist: Any
    edge_margin: float
    position_estimator: PositionEstimator = estimate_red_block_position
    last_route: str | None = field(default=None, init=False)
    detected_position: tuple[float, float, float] | None = field(
        default=None, init=False
    )

    def __post_init__(self) -> None:
        if self.primary.action_horizon != self.specialist.action_horizon:
            raise ValueError("gated policies must use the same action horizon")
        is_workspace_edge(
            (
                sum(WORKSPACE_X_BOUNDS_M) / 2,
                sum(WORKSPACE_Y_BOUNDS_M) / 2,
                0.0,
            ),
            margin=self.edge_margin,
        )

    @property
    def action_horizon(self) -> int:
        return int(self.primary.action_horizon)

    @property
    def device(self):
        return self.primary.device

    def reset(self, seed: int | None = None) -> None:
        self.last_route = None
        self.detected_position = None
        for policy in (self.primary, self.specialist):
            if hasattr(policy, "reset"):
                policy.reset(seed)

    def predict_chunk(self, rgb, depth, joint_position) -> np.ndarray:
        if self.last_route is None:
            self.detected_position = self.position_estimator(rgb, depth)
            if self.detected_position is not None and is_workspace_edge(
                self.detected_position,
                margin=self.edge_margin,
            ):
                self.last_route = "vision_expert"
                if hasattr(self.specialist, "set_target"):
                    self.specialist.set_target(self.detected_position)
            else:
                self.last_route = "learned_ensemble"
        policy = (
            self.specialist
            if self.last_route == "vision_expert"
            else self.primary
        )
        return policy.predict_chunk(rgb, depth, joint_position)

    def predict(self, rgb, depth, joint_position) -> np.ndarray:
        return self.predict_chunk(rgb, depth, joint_position)[0]
