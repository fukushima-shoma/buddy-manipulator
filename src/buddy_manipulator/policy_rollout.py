"""Closed-loop MuJoCo rollout helpers for learned manipulation policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Protocol

import numpy as np

from buddy_manipulator.dataset import CONTROLLED_JOINTS
from buddy_manipulator.grasping import evaluate_grasp, object_height
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import _mujoco


class RgbdPolicy(Protocol):
    def predict_chunk(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_position: np.ndarray,
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PolicyRolloutResult:
    success: bool
    steps: int
    duration: float
    initial_height: float
    final_height: float
    lift_delta: float
    max_lift_delta: float
    finger_contact_count: int
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def controlled_joint_positions(model: Any, data: Any) -> np.ndarray:
    mujoco = _mujoco()
    positions = []
    for name in CONTROLLED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if joint_id < 0:
            raise KeyError(f"unknown joint: {name}")
        positions.append(float(data.qpos[int(model.jnt_qposadr[joint_id])]))
    return np.asarray(positions, dtype=np.float32)


def rate_limit_action(
    model: Any,
    current_action: np.ndarray,
    requested_action: np.ndarray,
    *,
    max_arm_delta: float,
    max_gripper_delta: float,
) -> np.ndarray:
    if current_action.shape != (6,) or requested_action.shape != (6,):
        raise ValueError("actions must have shape (6,)")
    if max_arm_delta <= 0 or max_gripper_delta <= 0:
        raise ValueError("action delta limits must be positive")
    maximum_delta = np.asarray(
        [max_arm_delta] * 4 + [max_gripper_delta] * 2,
        dtype=np.float64,
    )
    limited = np.clip(
        requested_action,
        current_action - maximum_delta,
        current_action + maximum_delta,
    )
    control_range = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    return np.clip(limited, control_range[:, 0], control_range[:, 1])


def run_closed_loop_policy(
    model: Any,
    data: Any,
    policy: RgbdPolicy,
    camera: RgbdCamera,
    *,
    control_hz: float = 5.0,
    max_seconds: float = 8.0,
    success_hold_seconds: float = 0.4,
    max_arm_delta: float = 0.20,
    max_gripper_delta: float = 0.01,
    execute_chunk_steps: int = 1,
    viewer: Any | None = None,
    realtime: bool = False,
) -> PolicyRolloutResult:
    if (
        control_hz <= 0
        or max_seconds <= 0
        or success_hold_seconds <= 0
        or execute_chunk_steps <= 0
    ):
        raise ValueError("timing parameters must be positive")
    mujoco = _mujoco()
    initial_height = object_height(model, data)
    max_height = initial_height
    control_period = 1.0 / control_hz
    physics_steps = max(1, round(control_period / float(model.opt.timestep)))
    maximum_control_steps = max(1, math.ceil(max_seconds * control_hz))
    success_steps_required = max(1, math.ceil(success_hold_seconds * control_hz))
    consecutive_success_steps = 0
    completed_steps = 0
    final_grasp = evaluate_grasp(model, data, initial_height)
    trace = []

    stable_success = False
    while completed_steps < maximum_control_steps:
        if viewer is not None and not viewer.is_running():
            break
        frame = camera.capture(data)
        observed_joint_position = controlled_joint_positions(model, data)
        requested_actions = policy.predict_chunk(
            frame.rgb,
            frame.depth,
            observed_joint_position,
        )
        if requested_actions.ndim != 2 or requested_actions.shape[1] != 6:
            raise ValueError("policy action chunk must have shape (horizon, 6)")
        for requested_action in requested_actions[:execute_chunk_steps]:
            action = rate_limit_action(
                model,
                np.asarray(data.ctrl, dtype=np.float64).copy(),
                requested_action,
                max_arm_delta=max_arm_delta,
                max_gripper_delta=max_gripper_delta,
            )
            data.ctrl[:] = action
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                if realtime:
                    time.sleep(float(model.opt.timestep))
            completed_steps += 1
            current_height = object_height(model, data)
            max_height = max(max_height, current_height)
            final_grasp = evaluate_grasp(model, data, initial_height)
            trace.append(
                {
                    "step": completed_steps,
                    "time": completed_steps * control_period,
                    "policy_observation_joint_position": (
                        observed_joint_position.tolist()
                    ),
                    "requested_action": requested_action.tolist(),
                    "applied_action": action.tolist(),
                    "object_height": current_height,
                    "lift_delta": current_height - initial_height,
                    "finger_contact_count": final_grasp.finger_contact_count,
                }
            )
            if final_grasp.success:
                consecutive_success_steps += 1
                if consecutive_success_steps >= success_steps_required:
                    stable_success = True
                    break
            else:
                consecutive_success_steps = 0
            if completed_steps >= maximum_control_steps:
                break
        if stable_success:
            break

    return PolicyRolloutResult(
        success=stable_success,
        steps=completed_steps,
        duration=completed_steps * control_period,
        initial_height=initial_height,
        final_height=final_grasp.final_height,
        lift_delta=final_grasp.lift_delta,
        max_lift_delta=max_height - initial_height,
        finger_contact_count=final_grasp.finger_contact_count,
        trace=trace,
    )
