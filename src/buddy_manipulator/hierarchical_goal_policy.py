"""Hierarchical grasp-to-place execution for Phase 5F."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from buddy_manipulator.goal_task import (
    ManipulationGoal,
    detect_goal_object,
    execute_canonical_handoff,
    execute_goal_grasp,
    evaluate_goal_task,
    object_position,
)
from buddy_manipulator.goal_policy_rollout import run_closed_loop_goal_policy
from buddy_manipulator.policy_rollout import (
    controlled_joint_positions,
    rate_limit_action,
)
from buddy_manipulator.simulation import _mujoco
from buddy_manipulator.task_phase import GOAL_PHASE_DIM, encode_goal_phase


@dataclass
class SkillGoalPolicyRunner:
    """Apply the object-only or target-only goal contract at runtime."""

    policy: Any
    skill: str

    def __post_init__(self) -> None:
        if self.skill not in ("grasp", "place"):
            raise ValueError("skill must be grasp or place")

    @property
    def model(self) -> Any:
        return self.policy.model

    @property
    def action_horizon(self) -> int:
        return self.policy.action_horizon

    def predict_chunk(self, rgb, depth, joint_position, goal, phase=None):
        skill_goal = np.asarray(goal, dtype=np.float32).copy()
        if self.skill == "grasp":
            skill_goal[2:4] = 0.0
        elif str(getattr(self.policy, "goal_skill", "full")) != "handoff_place":
            skill_goal[0:2] = 0.0
        return self.policy.predict_chunk(
            rgb, depth, joint_position, skill_goal, phase
        )


@dataclass
class LiftTransitionGate:
    """Detect a stable grasp using only RGB-D height and gripper proprioception."""

    minimum_object_height_m: float = 0.09
    maximum_closed_finger_position: float = 0.012
    required_observations: int = 2
    _consecutive_observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.minimum_object_height_m <= 0:
            raise ValueError("minimum object height must be positive")
        if self.maximum_closed_finger_position < 0:
            raise ValueError("closed finger position must be non-negative")
        if self.required_observations <= 0:
            raise ValueError("required observations must be positive")

    def reset(self) -> None:
        self._consecutive_observations = 0

    def update(
        self,
        object_height_m: float | None,
        joint_position: np.ndarray,
    ) -> bool:
        if joint_position.shape != (6,):
            raise ValueError("joint_position must have shape (6,)")
        fingers_closed = bool(
            np.max(np.abs(joint_position[-2:]))
            <= self.maximum_closed_finger_position
        )
        object_lifted = bool(
            object_height_m is not None
            and np.isfinite(object_height_m)
            and object_height_m >= self.minimum_object_height_m
        )
        if fingers_closed and object_lifted:
            self._consecutive_observations += 1
        else:
            self._consecutive_observations = 0
        return self._consecutive_observations >= self.required_observations


@dataclass(frozen=True)
class HierarchicalGoalPolicyResult:
    success: bool
    steps: int
    placement_error_m: float
    selected_object_in_target: bool
    distractor_in_target: bool
    transitioned: bool
    transition_step: int | None
    transition_mode: str
    max_detected_object_height_m: float | None
    handoff_normalized: bool
    handoff_object_position_before_m: tuple[float, float, float] | None
    handoff_object_position_after_m: tuple[float, float, float] | None
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_expert_grasp_then_place_policy(
    model: Any,
    data: Any,
    place_policy: Any,
    camera: Any,
    goal: ManipulationGoal,
    *,
    grasp_pose_policy: Any | None = None,
    normalize_handoff: bool = False,
    handoff_position_m: tuple[float, float, float] = (0.20, 0.06, 0.20),
    handoff_duration_seconds: float = 1.2,
    control_hz: float = 5.0,
    max_seconds: float = 18.0,
) -> HierarchicalGoalPolicyResult:
    """Measure the learned place skill behind a deterministic RGB-D/IK grasp."""
    prelude_seconds = 5.9 + (handoff_duration_seconds if normalize_handoff else 0.0)
    if control_hz <= 0 or max_seconds <= prelude_seconds:
        raise ValueError("max seconds must leave time after the expert grasp")
    if normalize_handoff and handoff_duration_seconds <= 0:
        raise ValueError("handoff duration must be positive")
    initial_frame = camera.capture(data)
    detection = (
        grasp_pose_policy.predict_detection(initial_frame, goal)
        if grasp_pose_policy is not None
        else detect_goal_object(initial_frame, goal)
    )
    joint_history = [controlled_joint_positions(model, data)]
    next_sample_time = [float(data.time) + 1.0 / control_hz]

    def record_joint_history(_model: Any, callback_data: Any) -> None:
        if callback_data.time + 1e-9 < next_sample_time[0]:
            return
        joint_history.append(controlled_joint_positions(model, callback_data))
        next_sample_time[0] = float(callback_data.time) + 1.0 / control_hz

    execute_goal_grasp(
        model,
        data,
        detection,
        step_callback=record_joint_history,
    )
    handoff_before = tuple(
        float(value) for value in object_position(model, data, goal.object_color)
    )
    if normalize_handoff:
        execute_canonical_handoff(
            model,
            data,
            position_m=handoff_position_m,
            duration=handoff_duration_seconds,
            step_callback=record_joint_history,
        )
    handoff_after = tuple(
        float(value) for value in object_position(model, data, goal.object_color)
    )
    place_policy.reset()
    if hasattr(place_policy, "prime_joint_history"):
        place_policy.prime_joint_history(joint_history)
    post_grasp_height = None
    try:
        post_grasp_height = float(detect_goal_object(camera.capture(data), goal).z)
    except RuntimeError:
        pass
    place_result = run_closed_loop_goal_policy(
        model,
        data,
        SkillGoalPolicyRunner(place_policy, "place"),
        camera,
        goal,
        control_hz=control_hz,
        max_seconds=max_seconds - prelude_seconds,
        execute_chunk_steps=1,
    )
    expert_steps = math.ceil(prelude_seconds * control_hz)
    trace = [
        {**record, "step": int(record["step"]) + expert_steps, "skill": "place"}
        for record in place_result.trace
    ]
    return HierarchicalGoalPolicyResult(
        success=place_result.success,
        steps=expert_steps + place_result.steps,
        placement_error_m=place_result.placement_error_m,
        selected_object_in_target=place_result.selected_object_in_target,
        distractor_in_target=place_result.distractor_in_target,
        transitioned=True,
        transition_step=expert_steps,
        transition_mode=(
            str(getattr(grasp_pose_policy, "transition_mode", "residual_grasp"))
            if grasp_pose_policy is not None
            else "expert_grasp"
        ),
        max_detected_object_height_m=post_grasp_height,
        handoff_normalized=normalize_handoff,
        handoff_object_position_before_m=handoff_before,
        handoff_object_position_after_m=handoff_after,
        trace=trace,
    )


def run_hierarchical_goal_policy(
    model: Any,
    data: Any,
    grasp_policy: Any,
    place_policy: Any,
    camera: Any,
    goal: ManipulationGoal,
    *,
    transition_mode: str = "observable",
    oracle_transition_seconds: float = 4.7,
    control_hz: float = 5.0,
    max_seconds: float = 18.0,
    success_hold_seconds: float = 0.4,
    max_arm_delta: float = 0.20,
    max_gripper_delta: float = 0.01,
    lift_gate: LiftTransitionGate | None = None,
) -> HierarchicalGoalPolicyResult:
    """Execute grasp and place policies with an explicit skill transition."""
    if transition_mode not in ("observable", "oracle"):
        raise ValueError("transition mode must be observable or oracle")
    if (
        min(
            oracle_transition_seconds,
            control_hz,
            max_seconds,
            success_hold_seconds,
        )
        <= 0
    ):
        raise ValueError("timing parameters must be positive")
    for policy, expected_skill in (
        (grasp_policy, "grasp"),
        (place_policy, "place"),
    ):
        skill = str(getattr(policy, "goal_skill", "full"))
        if skill not in (expected_skill, "full"):
            raise ValueError(
                f"{expected_skill} checkpoint declares incompatible skill: {skill}"
            )

    mujoco = _mujoco()
    gate = lift_gate or LiftTransitionGate()
    gate.reset()
    grasp_policy.reset()
    place_policy.reset()
    control_period = 1.0 / control_hz
    physics_steps = max(1, round(control_period / float(model.opt.timestep)))
    maximum_steps = max(1, math.ceil(max_seconds * control_hz))
    success_steps_required = max(1, math.ceil(success_hold_seconds * control_hz))
    consecutive_successes = 0
    completed_steps = 0
    transitioned = False
    transition_step = None
    maximum_detected_height = None
    trace: list[dict[str, Any]] = []
    task_result = evaluate_goal_task(model, data, goal)

    while completed_steps < maximum_steps:
        frame = camera.capture(data)
        joint_position = controlled_joint_positions(model, data)
        detected_height = None
        try:
            detected_height = float(detect_goal_object(frame, goal).z)
            maximum_detected_height = (
                detected_height
                if maximum_detected_height is None
                else max(maximum_detected_height, detected_height)
            )
        except RuntimeError:
            pass

        if not transitioned:
            if transition_mode == "oracle":
                should_transition = completed_steps / control_hz >= oracle_transition_seconds
            else:
                should_transition = gate.update(detected_height, joint_position)
            if should_transition:
                transitioned = True
                transition_step = completed_steps
                place_policy.reset()

        active_skill = "place" if transitioned else "grasp"
        active_policy = place_policy if transitioned else grasp_policy
        skill_goal = goal.vector().copy()
        if active_skill == "grasp":
            skill_goal[2:4] = 0.0
        else:
            skill_goal[0:2] = 0.0
        phase = None
        if int(getattr(active_policy.model, "phase_dim", 0)):
            if active_policy.model.phase_dim != GOAL_PHASE_DIM:
                raise ValueError("checkpoint uses an unsupported task phase dimension")
            skill_start_step = transition_step if transitioned else 0
            phase = encode_goal_phase(
                (completed_steps - int(skill_start_step or 0)) / control_hz
            )
        actions = active_policy.predict_chunk(
            frame.rgb,
            frame.depth,
            joint_position,
            skill_goal,
            phase,
        )
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise ValueError("policy action chunk must have shape (horizon, 6)")
        requested_action = actions[0]
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
                "skill": active_skill,
                "detected_object_height_m": detected_height,
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
                break
        else:
            consecutive_successes = 0

    return HierarchicalGoalPolicyResult(
        success=bool(consecutive_successes >= success_steps_required),
        steps=completed_steps,
        placement_error_m=task_result.placement_error_m,
        selected_object_in_target=task_result.selected_object_in_target,
        distractor_in_target=task_result.distractor_in_target,
        transitioned=transitioned,
        transition_step=transition_step,
        transition_mode=transition_mode,
        max_detected_object_height_m=maximum_detected_height,
        handoff_normalized=False,
        handoff_object_position_before_m=None,
        handoff_object_position_after_m=None,
        trace=trace,
    )
