"""Run a learned behavior-cloning policy in MuJoCo closed loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from buddy_manipulator.behavior_cloning import BehaviorCloningRunner
from buddy_manipulator.collect_demos import sample_block_position, set_block_position
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.policy_rollout import run_closed_loop_policy
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a behavior-cloning policy in closed-loop MuJoCo."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--success-hold-seconds", type=float, default=0.4)
    parser.add_argument("--max-arm-delta", type=float, default=0.20)
    parser.add_argument("--max-gripper-delta", type=float, default=0.01)
    parser.add_argument(
        "--execute-chunk-steps",
        type=int,
        default=0,
        help="Actions executed before re-observation; 0 uses checkpoint horizon.",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase4/rollout_results.json"),
    )
    return parser.parse_args()


def run_episode(
    policy: BehaviorCloningRunner,
    block_position: tuple[float, float, float],
    args: argparse.Namespace,
):
    model, data = load_model()
    set_block_position(model, data, block_position)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
    execute_chunk_steps = (
        policy.action_horizon
        if args.execute_chunk_steps == 0
        else args.execute_chunk_steps
    )

    if args.headless:
        with RgbdCamera(model, width=args.width, height=args.height) as camera:
            return run_closed_loop_policy(
                model,
                data,
                policy,
                camera,
                control_hz=args.control_hz,
                max_seconds=args.max_seconds,
                success_hold_seconds=args.success_hold_seconds,
                max_arm_delta=args.max_arm_delta,
                max_gripper_delta=args.max_gripper_delta,
                execute_chunk_steps=execute_chunk_steps,
            )

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer, RgbdCamera(
        model,
        width=args.width,
        height=args.height,
    ) as camera:
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        result = run_closed_loop_policy(
            model,
            data,
            policy,
            camera,
            control_hz=args.control_hz,
            max_seconds=args.max_seconds,
            success_hold_seconds=args.success_hold_seconds,
            max_arm_delta=args.max_arm_delta,
            max_gripper_delta=args.max_gripper_delta,
            execute_chunk_steps=execute_chunk_steps,
            viewer=viewer,
            realtime=True,
        )
        print("Rollout complete. Close the MuJoCo window to continue.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        return result


def main() -> None:
    args = parse_args()
    if (
        args.episodes <= 0
        or args.width <= 0
        or args.height <= 0
        or args.execute_chunk_steps < 0
    ):
        raise ValueError("episodes and image dimensions must be positive")
    policy = BehaviorCloningRunner.from_checkpoint(
        args.checkpoint,
        device_name=args.device,
    )
    rng = random.Random(args.seed)
    episode_results = []
    print(f"policy device: {policy.device}", flush=True)
    for episode_index in range(args.episodes):
        block_position = sample_block_position(rng)
        result = run_episode(policy, block_position, args)
        record = {
            "episode": episode_index,
            "block_position_m": list(block_position),
            **result.to_dict(),
        }
        episode_results.append(record)
        print(
            f"episode {episode_index:03d}: success={str(result.success).lower()} "
            f"lift={result.lift_delta:.3f}m max_lift={result.max_lift_delta:.3f}m "
            f"steps={result.steps}",
            flush=True,
        )

    success_count = sum(bool(result["success"]) for result in episode_results)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(policy.device),
        "seed": args.seed,
        "episode_count": args.episodes,
        "action_horizon": policy.action_horizon,
        "execute_chunk_steps": (
            policy.action_horizon
            if args.execute_chunk_steps == 0
            else args.execute_chunk_steps
        ),
        "success_count": success_count,
        "success_rate": success_count / args.episodes,
        "mean_max_lift_delta_m": float(
            np.mean([result["max_lift_delta"] for result in episode_results])
        ),
        "episodes": episode_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"closed-loop success: {success_count}/{args.episodes} "
        f"({summary['success_rate']:.0%})",
        flush=True,
    )
    print(f"result JSON: {args.output}", flush=True)


if __name__ == "__main__":
    main()
