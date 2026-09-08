"""Evaluate baseline and residual RGB-D/IK grasp poses in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    ManipulationGoal,
    apply_object_positions,
    detect_goal_object,
    execute_goal_grasp,
    object_position,
    sample_object_positions,
)
from buddy_manipulator.grasp_pose_residual import GraspPoseResidualRunner
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--controller", choices=("baseline", "residual"), default="residual"
    )
    parser.add_argument("--blend", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3631)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/grasp_pose_residual/rollout_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("episodes and image dimensions must be positive")
    residual = GraspPoseResidualRunner.from_checkpoint(
        args.checkpoint, device_name=args.device, blend=args.blend
    )
    rng = random.Random(args.seed)
    episodes = []
    for index in range(args.episodes):
        object_color = OBJECT_COLORS[index % len(OBJECT_COLORS)]
        goal = ManipulationGoal(object_color, "green")
        positions = sample_object_positions(rng)
        model, data = load_model()
        apply_object_positions(model, data, positions)
        stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
        run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
        with RgbdCamera(model, width=args.width, height=args.height) as camera:
            frame = camera.capture(data)
            baseline = detect_goal_object(frame, goal)
            detection = (
                residual.predict_detection(frame, goal)
                if args.controller == "residual"
                else baseline
            )
            correction = (
                residual.last_residual_m.tolist()
                if args.controller == "residual"
                and residual.last_residual_m is not None
                else [0.0, 0.0, 0.0]
            )
            execute_goal_grasp(model, data, detection)
        selected = object_position(model, data, object_color)
        distractor_color = next(
            color for color in OBJECT_COLORS if color != object_color
        )
        distractor = object_position(model, data, distractor_color)
        success = bool(selected[2] >= 0.08 and distractor[2] <= 0.06)
        record = {
            "episode": index,
            "object_color": object_color,
            "object_start_positions_m": {
                color: list(position) for color, position in positions.items()
            },
            "baseline_detection_m": list(baseline.position),
            "applied_detection_m": list(detection.position),
            "residual_m": correction,
            "selected_final_position_m": selected.tolist(),
            "distractor_final_position_m": distractor.tolist(),
            "success": success,
        }
        episodes.append(record)
        print(
            f"episode {index:03d}: success={str(success).lower()} "
            f"object={object_color} height={selected[2]:.3f}m "
            f"residual=({correction[0] * 1000:.1f},"
            f"{correction[1] * 1000:.1f},{correction[2] * 1000:.1f})mm",
            flush=True,
        )
    successes = sum(record["success"] for record in episodes)
    per_object = {
        color: {
            "successes": sum(
                record["success"]
                for record in episodes
                if record["object_color"] == color
            ),
            "episodes": sum(
                record["object_color"] == color for record in episodes
            ),
        }
        for color in OBJECT_COLORS
    }
    summary = {
        "checkpoint": str(args.checkpoint),
        "controller": args.controller,
        "blend": residual.blend if args.controller == "residual" else 0.0,
        "seed": args.seed,
        "episode_count": args.episodes,
        "success_count": successes,
        "success_rate": successes / args.episodes,
        "per_object": per_object,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"grasp success: {successes}/{args.episodes}")
    print(f"result JSON: {args.output}")


if __name__ == "__main__":
    main()
