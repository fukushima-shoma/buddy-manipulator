"""Evaluate Phase 5F hierarchical goal policies in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from buddy_manipulator.behavior_cloning import BehaviorCloningRunner
from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    sample_object_positions,
)
from buddy_manipulator.hierarchical_goal_policy import (
    run_expert_grasp_then_place_policy,
    run_hierarchical_goal_policy,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.rollout_goal_bc import parse_goal
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical grasp and place BC policies."
    )
    parser.add_argument("grasp_checkpoint", type=Path)
    parser.add_argument("place_checkpoint", type=Path)
    parser.add_argument("--goal", type=parse_goal, default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3025)
    parser.add_argument(
        "--grasp-controller",
        choices=("learned", "expert"),
        default="learned",
        help="Use learned grasp or an RGB-D/IK diagnostic upper bound.",
    )
    parser.add_argument(
        "--transition",
        choices=("observable", "oracle"),
        default="observable",
    )
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/hierarchical/rollout_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("episodes and image dimensions must be positive")
    grasp_policy = BehaviorCloningRunner.from_checkpoint(
        args.grasp_checkpoint, device_name=args.device
    )
    place_policy = BehaviorCloningRunner.from_checkpoint(
        args.place_checkpoint, device_name=args.device
    )
    if grasp_policy.model.goal_dim != 4 or place_policy.model.goal_dim != 4:
        raise ValueError("both checkpoints must use four-value manipulation goals")

    goals = [
        ManipulationGoal(object_color, target_color)
        for object_color in OBJECT_COLORS
        for target_color in TARGET_COLORS
    ]
    rng = random.Random(args.seed)
    episodes = []
    for index in range(args.episodes):
        goal = args.goal or goals[index % len(goals)]
        positions = sample_object_positions(rng)
        model, data = load_model()
        apply_object_positions(model, data, positions)
        stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
        run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
        grasp_policy.reset(args.seed * 100_000 + index)
        place_policy.reset(args.seed * 100_000 + index)
        with RgbdCamera(model, width=args.width, height=args.height) as camera:
            if args.grasp_controller == "expert":
                result = run_expert_grasp_then_place_policy(
                    model,
                    data,
                    place_policy,
                    camera,
                    goal,
                    control_hz=args.control_hz,
                    max_seconds=args.max_seconds,
                )
            else:
                result = run_hierarchical_goal_policy(
                    model,
                    data,
                    grasp_policy,
                    place_policy,
                    camera,
                    goal,
                    transition_mode=args.transition,
                    control_hz=args.control_hz,
                    max_seconds=args.max_seconds,
                )
        record = {
            "episode": index,
            "goal": goal.to_dict(),
            "object_start_positions_m": {
                color: list(position) for color, position in positions.items()
            },
            **result.to_dict(),
        }
        episodes.append(record)
        print(
            f"episode {index:03d}: success={str(result.success).lower()} "
            f"goal={goal.object_color}->{goal.target_color} "
            f"transitioned={str(result.transitioned).lower()} "
            f"handoff={result.transition_step} error={result.placement_error_m:.3f}m",
            flush=True,
        )

    successes = sum(record["success"] for record in episodes)
    transitions = sum(record["transitioned"] for record in episodes)
    per_goal = {}
    for goal in goals:
        key = f"{goal.object_color}:{goal.target_color}"
        matching = [
            record
            for record in episodes
            if record["goal"]["object_color"] == goal.object_color
            and record["goal"]["target_color"] == goal.target_color
        ]
        if matching:
            per_goal[key] = {
                "successes": sum(record["success"] for record in matching),
                "episodes": len(matching),
                "transitions": sum(record["transitioned"] for record in matching),
            }
    summary = {
        "grasp_checkpoint": str(args.grasp_checkpoint),
        "place_checkpoint": str(args.place_checkpoint),
        "device": str(grasp_policy.device),
        "seed": args.seed,
        "grasp_controller": args.grasp_controller,
        "transition_mode": (
            "expert_grasp" if args.grasp_controller == "expert" else args.transition
        ),
        "fixed_goal": args.goal.to_dict() if args.goal else None,
        "episode_count": args.episodes,
        "success_count": successes,
        "success_rate": successes / args.episodes,
        "transition_count": transitions,
        "transition_rate": transitions / args.episodes,
        "per_goal": per_goal,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"hierarchical success: {successes}/{args.episodes}")
    print(f"skill handoffs: {transitions}/{args.episodes}")
    print(f"result JSON: {args.output}")


if __name__ == "__main__":
    main()
