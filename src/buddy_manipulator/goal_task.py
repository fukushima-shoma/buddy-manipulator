"""Goal-conditioned multi-object pick-and-place task primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Callable

import numpy as np

from buddy_manipulator.kinematics import Pose, inverse_kinematics
from buddy_manipulator.simulation import Keyframe, _mujoco, run_keyframes
from buddy_manipulator.vision import (
    WorldDetection,
    detect_colored_object,
    locate_detection_in_world,
)


OBJECT_BODIES = {"red": "red_block", "purple": "purple_block"}
TARGET_GEOMS = {"green": "goal_zone", "yellow": "yellow_goal_zone"}
OBJECT_COLORS = tuple(OBJECT_BODIES)
TARGET_COLORS = tuple(TARGET_GEOMS)


@dataclass(frozen=True)
class ManipulationGoal:
    object_color: str
    target_color: str

    def __post_init__(self) -> None:
        if self.object_color not in OBJECT_BODIES:
            raise ValueError(f"unsupported object color: {self.object_color}")
        if self.target_color not in TARGET_GEOMS:
            raise ValueError(f"unsupported target color: {self.target_color}")

    @property
    def object_body_name(self) -> str:
        return OBJECT_BODIES[self.object_color]

    @property
    def target_geom_name(self) -> str:
        return TARGET_GEOMS[self.target_color]

    @property
    def instruction(self) -> str:
        return f"place the {self.object_color} block in the {self.target_color} zone"

    def vector(self) -> np.ndarray:
        values = np.zeros(4, dtype=np.float32)
        values[OBJECT_COLORS.index(self.object_color)] = 1.0
        values[2 + TARGET_COLORS.index(self.target_color)] = 1.0
        return values

    def to_dict(self) -> dict[str, str]:
        return {
            "object_color": self.object_color,
            "target_color": self.target_color,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class GoalTaskResult:
    success: bool
    selected_object_in_target: bool
    distractor_in_target: bool
    placement_error_m: float
    selected_object_position_m: tuple[float, float, float]
    target_position_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_object_position(
    model: Any,
    data: Any,
    object_color: str,
    position: tuple[float, float, float],
) -> None:
    if object_color not in OBJECT_BODIES:
        raise ValueError(f"unsupported object color: {object_color}")
    mujoco = _mujoco()
    joint_name = f"{OBJECT_BODIES[object_color]}_free"
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"unknown joint: {joint_name}")
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def sample_object_positions(
    rng: random.Random,
    *,
    minimum_separation: float = 0.11,
) -> dict[str, tuple[float, float, float]]:
    """Sample two reachable, non-overlapping object placements."""
    if minimum_separation <= 0:
        raise ValueError("minimum separation must be positive")
    for _ in range(1_000):
        red = (rng.uniform(0.23, 0.34), rng.uniform(0.04, 0.13), 0.025)
        purple = (rng.uniform(0.23, 0.34), rng.uniform(0.04, 0.13), 0.025)
        if math.dist(red[:2], purple[:2]) >= minimum_separation:
            return {"red": red, "purple": purple}
    raise RuntimeError("could not sample separated object positions")


def apply_object_positions(
    model: Any,
    data: Any,
    positions: dict[str, tuple[float, float, float]],
) -> None:
    configure_goal_scene(model, data)
    for color in OBJECT_COLORS:
        set_object_position(model, data, color, positions[color])


def configure_goal_scene(model: Any, data: Any) -> None:
    """Move optional Phase 5 assets from storage into the active workspace."""
    mujoco = _mujoco()
    yellow_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "yellow_goal_zone_body"
    )
    if yellow_body < 0:
        raise KeyError("unknown body: yellow_goal_zone_body")
    model.body_pos[yellow_body] = (0.32, -0.16, 0.0)
    for geom_name in ("left_finger_pad", "right_finger_pad"):
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        )
        if geom_id < 0:
            raise KeyError(f"unknown geom: {geom_name}")
        model.geom_friction[geom_id][0] = 1.2
    mujoco.mj_forward(model, data)


def detect_goal_object(frame: Any, goal: ManipulationGoal) -> WorldDetection:
    pixel = detect_colored_object(frame.rgb, color=goal.object_color)
    if pixel is None:
        raise RuntimeError(f"{goal.object_color} block was not detected")
    return locate_detection_in_world(
        pixel,
        frame.depth,
        vertical_fov_degrees=frame.vertical_fov_degrees,
        camera_position=frame.camera_position,
        camera_rotation=frame.camera_rotation,
    )


def _top_down_joints(x: float, y: float, z: float):
    return inverse_kinematics(
        Pose(x=x, y=y, z=z, pitch=-math.pi / 2),
        elbow_up=True,
    )


def target_position(model: Any, data: Any, goal: ManipulationGoal) -> np.ndarray:
    mujoco = _mujoco()
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, goal.target_geom_name
    )
    if geom_id < 0:
        raise KeyError(f"unknown target geom: {goal.target_geom_name}")
    return np.asarray(data.geom_xpos[geom_id], dtype=np.float64).copy()


def goal_grasp_keyframes(detection: WorldDetection) -> list[Keyframe]:
    """Build the goal-independent acquisition skill through safe retraction."""
    pickup_angle = math.atan2(detection.y, detection.x)
    transport_radius = 0.20
    pickup_transport = (
        transport_radius * math.cos(pickup_angle),
        transport_radius * math.sin(pickup_angle),
    )
    grasp_height = detection.z - 0.012
    grasp = _top_down_joints(detection.x, detection.y, grasp_height)
    return [
        Keyframe(
            1.2,
            _top_down_joints(detection.x, detection.y, detection.z + 0.10),
            0.03,
        ),
        Keyframe(1.0, grasp, 0.03),
        Keyframe(1.0, grasp, 0.0),
        Keyframe(1.5, _top_down_joints(detection.x, detection.y, 0.20), 0.0),
        Keyframe(1.2, _top_down_joints(*pickup_transport, 0.20), 0.0),
    ]


def execute_goal_grasp(
    model: Any,
    data: Any,
    detection: WorldDetection,
    *,
    viewer: Any | None = None,
    step_callback: Callable[[Any, Any], None] | None = None,
) -> None:
    """Execute the deterministic Phase 5 grasp skill only."""
    run_keyframes(
        model,
        data,
        goal_grasp_keyframes(detection),
        viewer=viewer,
        realtime=viewer is not None,
        step_callback=step_callback,
    )


def canonical_handoff_keyframe(
    *,
    position_m: tuple[float, float, float] = (0.20, 0.06, 0.20),
    duration: float = 1.2,
) -> Keyframe:
    """Move a held object to one reproducible pre-placement arm pose."""
    if duration <= 0:
        raise ValueError("handoff duration must be positive")
    if position_m[2] <= 0:
        raise ValueError("handoff height must be positive")
    return Keyframe(duration, _top_down_joints(*position_m), 0.0)


def execute_canonical_handoff(
    model: Any,
    data: Any,
    *,
    position_m: tuple[float, float, float] = (0.20, 0.06, 0.20),
    duration: float = 1.2,
    viewer: Any | None = None,
    step_callback: Callable[[Any, Any], None] | None = None,
) -> None:
    """Normalize the closed-gripper state before learned placement."""
    run_keyframes(
        model,
        data,
        [canonical_handoff_keyframe(position_m=position_m, duration=duration)],
        viewer=viewer,
        realtime=viewer is not None,
        step_callback=step_callback,
    )


def goal_place_keyframes(
    model: Any,
    data: Any,
    goal: ManipulationGoal,
    *,
    release_residual_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[Keyframe]:
    """Build deterministic transport and release motion for a held object."""
    residual = np.asarray(release_residual_m, dtype=np.float64)
    if residual.shape != (3,):
        raise ValueError("release residual must have three values")
    destination = target_position(model, data, goal) + residual
    target_x, target_y = float(destination[0]), float(destination[1])
    target_angle = math.atan2(target_y, target_x)
    transport_radius = 0.20
    target_transport = (
        transport_radius * math.cos(target_angle),
        transport_radius * math.sin(target_angle),
    )
    place_height = 0.052 + float(residual[2])
    return [
        Keyframe(2.5, _top_down_joints(*target_transport, 0.20), 0.0),
        Keyframe(1.5, _top_down_joints(target_x, target_y, 0.20), 0.0),
        Keyframe(1.5, _top_down_joints(target_x, target_y, place_height), 0.0),
        Keyframe(1.0, _top_down_joints(target_x, target_y, place_height), 0.03),
        Keyframe(1.0, _top_down_joints(target_x, target_y, 0.16), 0.03),
    ]


def execute_goal_place(
    model: Any,
    data: Any,
    goal: ManipulationGoal,
    *,
    release_residual_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    viewer: Any | None = None,
    step_callback: Callable[[Any, Any], None] | None = None,
) -> GoalTaskResult:
    """Transport and release an already-held object using deterministic IK."""
    run_keyframes(
        model,
        data,
        goal_place_keyframes(
            model,
            data,
            goal,
            release_residual_m=release_residual_m,
        ),
        viewer=viewer,
        realtime=viewer is not None,
        step_callback=step_callback,
    )
    return evaluate_goal_task(model, data, goal)


def execute_pick_and_place(
    model: Any,
    data: Any,
    goal: ManipulationGoal,
    detection: WorldDetection,
    *,
    normalize_handoff: bool = False,
    handoff_position_m: tuple[float, float, float] = (0.20, 0.06, 0.20),
    handoff_duration_seconds: float = 1.2,
    viewer: Any | None = None,
    step_callback: Callable[[Any, Any], None] | None = None,
) -> GoalTaskResult:
    """Pick the selected object and release it over the requested target."""
    handoff_keyframes = (
        [
            canonical_handoff_keyframe(
                position_m=handoff_position_m,
                duration=handoff_duration_seconds,
            )
        ]
        if normalize_handoff
        else []
    )
    keyframes = [
        *goal_grasp_keyframes(detection),
        *handoff_keyframes,
        *goal_place_keyframes(model, data, goal),
    ]
    run_keyframes(
        model,
        data,
        keyframes,
        viewer=viewer,
        realtime=viewer is not None,
        step_callback=step_callback,
    )
    return evaluate_goal_task(model, data, goal)


def _body_position(model: Any, data: Any, body_name: str) -> np.ndarray:
    mujoco = _mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"unknown body: {body_name}")
    return np.asarray(data.xpos[body_id], dtype=np.float64).copy()


def object_position(model: Any, data: Any, object_color: str) -> np.ndarray:
    """Return one object position for simulation-only evaluation."""
    if object_color not in OBJECT_BODIES:
        raise ValueError(f"unsupported object color: {object_color}")
    return _body_position(model, data, OBJECT_BODIES[object_color])


def evaluate_goal_task(
    model: Any,
    data: Any,
    goal: ManipulationGoal,
) -> GoalTaskResult:
    destination = target_position(model, data, goal)
    selected = _body_position(model, data, goal.object_body_name)
    distractor_color = next(color for color in OBJECT_COLORS if color != goal.object_color)
    distractor = _body_position(model, data, OBJECT_BODIES[distractor_color])
    mujoco = _mujoco()
    target_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, goal.target_geom_name
    )
    target_radius = float(model.geom_size[target_id][0])
    selected_error = float(np.linalg.norm(selected[:2] - destination[:2]))
    distractor_error = float(np.linalg.norm(distractor[:2] - destination[:2]))
    selected_inside = selected_error <= target_radius - 0.01 and selected[2] <= 0.06
    distractor_inside = distractor_error <= target_radius
    return GoalTaskResult(
        success=bool(selected_inside and not distractor_inside),
        selected_object_in_target=bool(selected_inside),
        distractor_in_target=bool(distractor_inside),
        placement_error_m=selected_error,
        selected_object_position_m=tuple(float(value) for value in selected),
        target_position_m=tuple(float(value) for value in destination),
    )
