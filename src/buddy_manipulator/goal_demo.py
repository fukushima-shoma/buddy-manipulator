"""Run one visible or headless goal-conditioned pick-and-place task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    detect_goal_object,
    execute_pick_and_place,
    sample_object_positions,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a goal-conditioned multi-object pick-and-place demo."
    )
    parser.add_argument("--object", choices=OBJECT_COLORS, default="red")
    parser.add_argument("--target", choices=TARGET_COLORS, default="green")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/goal_demo.json"),
    )
    return parser.parse_args()


def save_result(args, goal, positions, result) -> None:
    payload = {
        "goal": goal.to_dict(),
        "object_start_positions_m": {
            color: list(position) for color, position in positions.items()
        },
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"goal success: {str(result.success).lower()} "
        f"placement error={result.placement_error_m:.3f}m"
    )
    print(f"result JSON: {args.output}")


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    goal = ManipulationGoal(args.object, args.target)
    positions = sample_object_positions(random.Random(args.seed))
    model, data = load_model()
    apply_object_positions(model, data, positions)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
    print(f"instruction: {goal.instruction}", flush=True)

    if args.headless:
        with RgbdCamera(model, width=args.width, height=args.height) as camera:
            detection = detect_goal_object(camera.capture(data), goal)
            result = execute_pick_and_place(model, data, goal, detection)
        save_result(args, goal, positions, result)
        if not result.success:
            raise SystemExit(1)
        return

    import mujoco.viewer

    print("MuJoCo viewer started. Executing goal task...", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer, RgbdCamera(
        model, width=args.width, height=args.height
    ) as camera:
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        detection = detect_goal_object(camera.capture(data), goal)
        result = execute_pick_and_place(
            model,
            data,
            goal,
            detection,
            viewer=viewer,
        )
        save_result(args, goal, positions, result)
        print("Close the MuJoCo window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
