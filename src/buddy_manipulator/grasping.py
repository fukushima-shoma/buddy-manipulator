"""Perception-driven scripted grasping and success evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from buddy_manipulator.kinematics import Pose, inverse_kinematics
from buddy_manipulator.simulation import Keyframe, _mujoco, run_keyframes
from buddy_manipulator.vision import WorldDetection


@dataclass(frozen=True)
class GraspResult:
    success: bool
    initial_height: float
    final_height: float
    lift_delta: float
    finger_contact_count: int


def object_height(model: Any, data: Any, body_name: str = "red_block") -> float:
    mujoco = _mujoco()
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise KeyError(f"unknown body: {body_name}")
    return float(data.xpos[body_id][2])


def _finger_contact_count(
    model: Any,
    data: Any,
    object_body_name: str,
) -> int:
    mujoco = _mujoco()
    object_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
    )
    finger_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
    }
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        }
        if object_body in bodies and bodies.intersection(finger_bodies):
            count += 1
    return count


def _top_down_joints(x: float, y: float, z: float):
    return inverse_kinematics(
        Pose(x=x, y=y, z=z, pitch=-math.pi / 2),
        elbow_up=True,
    )


def evaluate_grasp(
    model: Any,
    data: Any,
    initial_height: float,
    *,
    object_body_name: str = "red_block",
) -> GraspResult:
    final_height = object_height(model, data, object_body_name)
    contacts = _finger_contact_count(model, data, object_body_name)
    lift_delta = final_height - initial_height
    return GraspResult(
        success=lift_delta >= 0.05 and contacts > 0,
        initial_height=initial_height,
        final_height=final_height,
        lift_delta=lift_delta,
        finger_contact_count=contacts,
    )


def execute_grasp(
    model: Any,
    data: Any,
    detection: WorldDetection,
    *,
    object_body_name: str = "red_block",
    viewer: Any | None = None,
    step_callback: Callable[[Any, Any], None] | None = None,
) -> GraspResult:
    """Approach, descend, close, and lift using the perceived block position."""
    initial_height = object_height(model, data, object_body_name)
    pregrasp = _top_down_joints(
        detection.x,
        detection.y,
        detection.z + 0.10,
    )
    grasp = _top_down_joints(detection.x, detection.y, detection.z)
    lift = _top_down_joints(
        detection.x,
        detection.y,
        detection.z + 0.15,
    )
    realtime = viewer is not None
    run_keyframes(
        model,
        data,
        [
            Keyframe(1.5, pregrasp, 0.03),
            Keyframe(1.2, grasp, 0.03),
            Keyframe(1.0, grasp, 0.0),
            Keyframe(1.5, lift, 0.0),
        ],
        viewer=viewer,
        realtime=realtime,
        step_callback=step_callback,
    )

    return evaluate_grasp(
        model,
        data,
        initial_height,
        object_body_name=object_body_name,
    )
