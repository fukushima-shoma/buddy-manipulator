"""Closed-loop rollout for goal-conditioned manipulation policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from buddy_manipulator.goal_task import ManipulationGoal, evaluate_goal_task
from buddy_manipulator.policy_rollout import controlled_joint_positions, rate_limit_action
from buddy_manipulator.simulation import _mujoco


@dataclass(frozen=True)
class GoalPolicyRolloutResult:
    success: bool
    steps: int
    placement_error_m: float
    selected_object_in_target: bool
    distractor_in_target: bool
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_closed_loop_goal_policy(
    model: Any,
    data: Any,
    policy: Any,
    camera: Any,
    goal: ManipulationGoal,
    *,
    control_hz: float = 5.0,
    max_seconds: float = 18.0,
    success_hold_seconds: float = 0.4,
    max_arm_delta: float = 0.20,
    max_gripper_delta: float = 0.01,
    execute_chunk_steps: int = 0,
) -> GoalPolicyRolloutResult:
    if control_hz <= 0 or max_seconds <= 0 or success_hold_seconds <= 0:
        raise ValueError("timing parameters must be positive")
    chunk_steps = policy.action_horizon if execute_chunk_steps == 0 else execute_chunk_steps
    if chunk_steps <= 0:
        raise ValueError("execute chunk steps must be positive")
    mujoco = _mujoco()
    control_period = 1.0 / control_hz
    physics_steps = max(1, round(control_period / float(model.opt.timestep)))
    maximum_steps = max(1, math.ceil(max_seconds * control_hz))
    success_steps_required = max(1, math.ceil(success_hold_seconds * control_hz))
    consecutive_successes = 0
    completed_steps = 0
    trace = []
    task_result = evaluate_goal_task(model, data, goal)

    while completed_steps < maximum_steps:
        frame = camera.capture(data)
        joint_position = controlled_joint_positions(model, data)
        actions = policy.predict_chunk(
            frame.rgb,
            frame.depth,
            joint_position,
            goal.vector(),
        )
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise ValueError("policy action chunk must have shape (horizon, 6)")
        for requested_action in actions[:chunk_steps]:
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
            completed_steps += 1
            task_result = evaluate_goal_task(model, data, goal)
            trace.append(
                {
                    "step": completed_steps,
                    "requested_action": requested_action.tolist(),
                    "applied_action": action.tolist(),
                    "placement_error_m": task_result.placement_error_m,
                    "selected_object_in_target": (
                        task_result.selected_object_in_target
                    ),
                    "distractor_in_target": task_result.distractor_in_target,
                }
            )
            if task_result.success:
                consecutive_successes += 1
                if consecutive_successes >= success_steps_required:
                    return GoalPolicyRolloutResult(
                        success=True,
                        steps=completed_steps,
                        placement_error_m=task_result.placement_error_m,
                        selected_object_in_target=True,
                        distractor_in_target=False,
                        trace=trace,
                    )
            else:
                consecutive_successes = 0
            if completed_steps >= maximum_steps:
                break

    return GoalPolicyRolloutResult(
        success=False,
        steps=completed_steps,
        placement_error_m=task_result.placement_error_m,
        selected_object_in_target=task_result.selected_object_in_target,
        distractor_in_target=task_result.distractor_in_target,
        trace=trace,
    )
