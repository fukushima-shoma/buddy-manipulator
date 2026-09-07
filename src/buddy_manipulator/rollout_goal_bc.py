"""Evaluate a goal-conditioned BC checkpoint in closed-loop MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from buddy_manipulator.behavior_cloning import BehaviorCloningRunner
from buddy_manipulator.goal_policy_rollout import run_closed_loop_goal_policy
from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    sample_object_positions,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_goal(value: str) -> ManipulationGoal:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("goal must use OBJECT:TARGET")
    try:
        return ManipulationGoal(parts[0], parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate goal-conditioned behavior cloning in MuJoCo."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--goal", type=parse_goal, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1106)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=18.0)
    parser.add_argument("--execute-chunk-steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/goal_bc/rollout_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("episodes and image dimensions must be positive")
    policy = BehaviorCloningRunner.from_checkpoint(
        args.checkpoint, device_name=args.device
    )
    if policy.model.goal_dim != 4:
        raise ValueError("checkpoint is not a four-value goal-conditioned policy")
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
        with RgbdCamera(model, width=args.width, height=args.height) as camera:
            result = run_closed_loop_goal_policy(
                model,
                data,
                policy,
                camera,
                goal,
                control_hz=args.control_hz,
                max_seconds=args.max_seconds,
                execute_chunk_steps=args.execute_chunk_steps,
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
            f"error={result.placement_error_m:.3f}m steps={result.steps}",
            flush=True,
        )
    successes = sum(episode["success"] for episode in episodes)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(policy.device),
        "seed": args.seed,
        "fixed_goal": args.goal.to_dict() if args.goal else None,
        "episode_count": args.episodes,
        "success_count": successes,
        "success_rate": successes / args.episodes,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"goal-conditioned success: {successes}/{args.episodes}")
    print(f"result JSON: {args.output}")


if __name__ == "__main__":
    main()
