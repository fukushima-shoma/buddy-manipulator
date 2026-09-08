"""Evaluate critic-ranked grasp plus structured placement end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from buddy_manipulator.calibration_bias import OnlinePlanarBiasEstimator
from buddy_manipulator.domain_randomization import (
    DomainRandomizationConfig,
    apply_domain_randomization,
    sample_domain_randomization,
)
from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    detect_goal_object,
    evaluate_goal_task,
    execute_goal_grasp,
    execute_goal_place,
    goal_place_keyframes,
    object_position,
    sample_object_positions,
    target_position,
)
from buddy_manipulator.grasp_success_critic import (
    GraspSuccessCriticRunner,
    OutcomeRankedGraspPosePolicy,
)
from buddy_manipulator.kinematics import JointAngles, UnreachableTargetError
from buddy_manipulator.placement_success_critic import (
    PlacementSuccessCriticRunner,
    grid_release_residuals,
)
from buddy_manipulator.policy_rollout import controlled_joint_positions
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grasp_critic_checkpoint", type=Path)
    parser.add_argument("placement_critic_checkpoint", type=Path)
    parser.add_argument("--place-controller", choices=("baseline", "critic"), default="critic")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4641)
    parser.add_argument("--place-bias-x-mm", type=float, default=0.0)
    parser.add_argument("--place-bias-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--bias-knowledge",
        choices=("known", "online"),
        default="known",
        help="Supply the injected bias or estimate it from prior RGB-D outcomes.",
    )
    parser.add_argument("--bias-smoothing", type=float, default=0.5)
    parser.add_argument(
        "--bias-estimator-scope",
        choices=("global", "goal"),
        default="global",
        help="Share one estimate or retain separate estimates per object-target goal.",
    )
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--friction-scale-min", type=float, default=0.7)
    parser.add_argument("--friction-scale-max", type=float, default=1.3)
    parser.add_argument("--mass-scale-min", type=float, default=0.6)
    parser.add_argument("--mass-scale-max", type=float, default=1.6)
    parser.add_argument("--appearance-scale-min", type=float, default=0.7)
    parser.add_argument("--appearance-scale-max", type=float, default=1.15)
    parser.add_argument("--place-bias-jitter-mm", type=float, default=15.0)
    parser.add_argument(
        "--retry-failed-grasp",
        action="store_true",
        help="Verify observable lift and make one fresh RGB-D grasp attempt on failure.",
    )
    parser.add_argument(
        "--maximum-place-joint-step-rad",
        type=float,
        default=0.0,
        help="Interpolate placement targets at this maximum joint-space step; zero disables it.",
    )
    parser.add_argument("--minimum-score-improvement", type=float, default=0.03)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/structured/rollout_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    grasp_policy = OutcomeRankedGraspPosePolicy(
        GraspSuccessCriticRunner.from_checkpoint(
            args.grasp_critic_checkpoint, device_name=args.device
        )
    )
    place_critic = PlacementSuccessCriticRunner.from_checkpoint(
        args.placement_critic_checkpoint, device_name=args.device
    )
    bias = np.asarray([args.place_bias_x_mm, args.place_bias_y_mm], dtype=np.float32)
    randomization_config = DomainRandomizationConfig(
        finger_friction_scale=(args.friction_scale_min, args.friction_scale_max),
        object_mass_scale=(args.mass_scale_min, args.mass_scale_max),
        appearance_scale=(args.appearance_scale_min, args.appearance_scale_max),
        place_bias_jitter_mm=args.place_bias_jitter_mm,
    )
    command_candidates = grid_release_residuals(np.zeros(2, dtype=np.float32))
    estimators: dict[str, OnlinePlanarBiasEstimator] = {}
    rng = random.Random(args.seed)
    domain_rng = random.Random(args.seed + 1_000_003)
    goals = [
        ManipulationGoal(object_color, target_color)
        for object_color in OBJECT_COLORS
        for target_color in TARGET_COLORS
    ]
    records = []
    for index in range(args.episodes):
        goal = goals[index % len(goals)]
        estimator_key = (
            "global"
            if args.bias_estimator_scope == "global"
            else f"{goal.object_color}:{goal.target_color}"
        )
        estimator = estimators.setdefault(
            estimator_key,
            OnlinePlanarBiasEstimator(smoothing=args.bias_smoothing),
        )
        positions = sample_object_positions(rng)
        model, data = load_model()
        apply_object_positions(model, data, positions)
        domain_sample = None
        episode_bias = bias.astype(np.float64)
        if args.domain_randomization:
            domain_sample = sample_domain_randomization(
                domain_rng, randomization_config
            )
            apply_domain_randomization(model, data, domain_sample)
            episode_bias = episode_bias + np.asarray(
                domain_sample.place_bias_jitter_xy_mm, dtype=np.float64
            )
        run_keyframes(
            model,
            data,
            [Keyframe(0.7, JointAngles(-1.2, 0.6, -1.2, 0.6), 0.03)],
        )
        with RgbdCamera(model, width=160, height=120) as camera:
            detection = grasp_policy.predict_detection(camera.capture(data), goal)
            execute_goal_grasp(model, data, detection)
            grasp_attempts = 1
            grasp_retry_error = None
            grasp_lift_m = float(
                object_position(model, data, goal.object_color)[2]
                - positions[goal.object_color][2]
            )
            if args.retry_failed_grasp and grasp_lift_m < 0.04:
                try:
                    retry_detection = grasp_policy.predict_detection(
                        camera.capture(data), goal
                    )
                    execute_goal_grasp(model, data, retry_detection)
                    grasp_attempts += 1
                    grasp_lift_m = float(
                        object_position(model, data, goal.object_color)[2]
                        - positions[goal.object_color][2]
                    )
                except (RuntimeError, UnreachableTargetError) as exc:
                    grasp_retry_error = str(exc)
            bias_estimate_before = estimator.estimate_mm.copy()
            if args.place_controller == "critic":
                if args.bias_knowledge == "known":
                    predicted_candidates = command_candidates + episode_bias[None]
                    predicted_fallback = episode_bias
                else:
                    predicted_candidates = estimator.predicted_outcomes(
                        command_candidates
                    )
                    predicted_fallback = estimator.estimate_mm.copy()
                reachable_candidates = []
                reachable_commands = []
                for command, candidate in zip(
                    command_candidates, predicted_candidates
                ):
                    try:
                        goal_place_keyframes(
                            model,
                            data,
                            goal,
                            release_residual_m=(
                                float(candidate[0] / 1000.0),
                                float(candidate[1] / 1000.0),
                                0.0,
                            ),
                        )
                    except UnreachableTargetError:
                        continue
                    reachable_candidates.append(candidate)
                    reachable_commands.append(command)
                reachable_candidates = np.asarray(
                    reachable_candidates, dtype=np.float32
                )
                residual_mm, probabilities = place_critic.choose_residual(
                    controlled_joint_positions(model, data),
                    goal,
                    target_position(model, data, goal),
                    reachable_candidates,
                    fallback_xy_mm=predicted_fallback,
                    minimum_score_improvement=args.minimum_score_improvement,
                )
                selected_probability = float(
                    probabilities[
                        np.flatnonzero(
                            np.all(
                                np.isclose(reachable_candidates, residual_mm[None]),
                                axis=1,
                            )
                        )[0]
                    ]
                )
                selected_index = int(
                    np.flatnonzero(
                        np.all(
                            np.isclose(reachable_candidates, residual_mm[None]),
                            axis=1,
                        )
                    )[0]
                )
                selected_command_mm = np.asarray(
                    reachable_commands[selected_index], dtype=np.float64
                )
                actual_residual_mm = episode_bias + selected_command_mm
            else:
                residual_mm = episode_bias
                selected_command_mm = np.zeros(2, dtype=np.float64)
                actual_residual_mm = residual_mm
                selected_probability = None
            execution_error = None
            placement_executed = False
            if grasp_lift_m < 0.04:
                result = evaluate_goal_task(model, data, goal)
                execution_error = "grasp verification failed"
            else:
                try:
                    result = execute_goal_place(
                        model,
                        data,
                        goal,
                        release_residual_m=(
                            float(actual_residual_mm[0] / 1000.0),
                            float(actual_residual_mm[1] / 1000.0),
                            0.0,
                        ),
                        maximum_joint_step_rad=(
                            args.maximum_place_joint_step_rad
                            if args.maximum_place_joint_step_rad > 0.0
                            else None
                        ),
                    )
                    placement_executed = True
                except UnreachableTargetError as exc:
                    result = evaluate_goal_task(model, data, goal)
                    execution_error = str(exc)
            observed_position = None
            bias_estimate_after = estimator.estimate_mm.copy()
            if args.bias_knowledge == "online" and placement_executed:
                try:
                    observed = detect_goal_object(camera.capture(data), goal)
                    observed_position = list(observed.position)
                    bias_estimate_after = estimator.update(
                        np.asarray(observed.position[:2]),
                        target_position(model, data, goal)[:2],
                        selected_command_mm,
                    )
                except RuntimeError:
                    pass
        records.append(
            {
                "episode": index,
                "goal": goal.to_dict(),
                "object_start_positions_m": {
                    color: list(position) for color, position in positions.items()
                },
                "domain_randomization": (
                    None if domain_sample is None else domain_sample.to_dict()
                ),
                "episode_place_bias_xy_mm": episode_bias.tolist(),
                "grasp_attempts": grasp_attempts,
                "grasp_lift_m": grasp_lift_m,
                "grasp_retry_error": grasp_retry_error,
                "placement_executed": placement_executed,
                "execution_error": execution_error,
                "predicted_release_residual_xy_mm": residual_mm.tolist(),
                "selected_command_xy_mm": selected_command_mm.tolist(),
                "actual_release_residual_xy_mm": actual_residual_mm.tolist(),
                "selected_probability": selected_probability,
                "bias_estimate_before_mm": bias_estimate_before.tolist(),
                "bias_estimate_after_mm": bias_estimate_after.tolist(),
                "observed_final_object_position_m": observed_position,
                **result.to_dict(),
            }
        )
        print(
            f"episode {index:03d}: success={str(result.success).lower()} "
            f"goal={goal.object_color}->{goal.target_color} "
            f"residual=({residual_mm[0]:.1f},{residual_mm[1]:.1f})mm "
            f"error={result.placement_error_m:.3f}m",
            flush=True,
        )
    successes = sum(int(record["success"]) for record in records)
    output = {
        "format_version": 1,
        "controller": args.place_controller,
        "seed": args.seed,
        "episode_count": args.episodes,
        "success_count": successes,
        "success_rate": successes / args.episodes,
        "injected_place_bias_xy_mm": bias.tolist(),
        "domain_randomization_enabled": args.domain_randomization,
        "maximum_place_joint_step_rad": args.maximum_place_joint_step_rad,
        "domain_randomization_config": (
            randomization_config.to_dict() if args.domain_randomization else None
        ),
        "bias_knowledge": args.bias_knowledge,
        "bias_smoothing": args.bias_smoothing,
        "bias_estimator_scope": args.bias_estimator_scope,
        "final_bias_estimate_mm": (
            estimators["global"].estimate_mm.tolist()
            if "global" in estimators
            else None
        ),
        "final_bias_estimates_mm": {
            key: value.estimate_mm.tolist() for key, value in estimators.items()
        },
        "bias_observations": sum(value.observations for value in estimators.values()),
        "bias_assumption": (
            "the injected bias is supplied to candidate construction"
            if args.bias_knowledge == "known"
            else (
                "bias starts at zero and is estimated from prior RGB-D outcomes "
                f"with {args.bias_estimator_scope} scope"
            )
        ),
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"structured task success: {successes}/{args.episodes}")
    print(f"result JSON: {args.output}")


if __name__ == "__main__":
    main()
