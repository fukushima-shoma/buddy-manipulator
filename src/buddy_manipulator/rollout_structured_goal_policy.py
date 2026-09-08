"""Evaluate critic-ranked grasp plus structured placement end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    execute_goal_grasp,
    execute_goal_place,
    goal_place_keyframes,
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
    candidates = grid_release_residuals(bias)
    rng = random.Random(args.seed)
    goals = [
        ManipulationGoal(object_color, target_color)
        for object_color in OBJECT_COLORS
        for target_color in TARGET_COLORS
    ]
    records = []
    for index in range(args.episodes):
        goal = goals[index % len(goals)]
        positions = sample_object_positions(rng)
        model, data = load_model()
        apply_object_positions(model, data, positions)
        run_keyframes(
            model,
            data,
            [Keyframe(0.7, JointAngles(-1.2, 0.6, -1.2, 0.6), 0.03)],
        )
        with RgbdCamera(model, width=160, height=120) as camera:
            detection = grasp_policy.predict_detection(camera.capture(data), goal)
            execute_goal_grasp(model, data, detection)
            if args.place_controller == "critic":
                reachable_candidates = []
                for candidate in candidates:
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
                reachable_candidates = np.asarray(
                    reachable_candidates, dtype=np.float32
                )
                residual_mm, probabilities = place_critic.choose_residual(
                    controlled_joint_positions(model, data),
                    goal,
                    target_position(model, data, goal),
                    reachable_candidates,
                    fallback_xy_mm=bias,
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
            else:
                residual_mm = bias.astype(np.float64)
                selected_probability = None
            result = execute_goal_place(
                model,
                data,
                goal,
                release_residual_m=(
                    float(residual_mm[0] / 1000.0),
                    float(residual_mm[1] / 1000.0),
                    0.0,
                ),
            )
        records.append(
            {
                "episode": index,
                "goal": goal.to_dict(),
                "object_start_positions_m": {
                    color: list(position) for color, position in positions.items()
                },
                "release_residual_xy_mm": residual_mm.tolist(),
                "selected_probability": selected_probability,
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
        "bias_assumption": "the injected bias is known when critic candidates are constructed",
        "episodes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"structured task success: {successes}/{args.episodes}")
    print(f"result JSON: {args.output}")


if __name__ == "__main__":
    main()
