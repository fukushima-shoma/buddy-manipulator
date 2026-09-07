"""Run a learned behavior-cloning policy in MuJoCo closed loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from buddy_manipulator.behavior_cloning import BehaviorCloningRunner
from buddy_manipulator.collect_demos import sample_block_position, set_block_position
from buddy_manipulator.hybrid_policy import (
    ScriptedVisionGraspPolicyRunner,
    SpatialGatedPolicyRunner,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.policy_rollout import run_closed_loop_policy
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


@dataclass
class AveragingPolicyRunner:
    """Average action chunks from policies with the same horizon."""

    policies: list[Any]

    def __post_init__(self) -> None:
        if len(self.policies) < 2:
            raise ValueError("an ensemble needs at least two policies")
        horizons = {policy.action_horizon for policy in self.policies}
        if len(horizons) != 1:
            raise ValueError("ensemble policies must use the same action horizon")

    @property
    def action_horizon(self) -> int:
        return int(self.policies[0].action_horizon)

    @property
    def device(self):
        return self.policies[0].device

    def reset(self, seed: int | None = None) -> None:
        for policy in self.policies:
            if hasattr(policy, "reset"):
                policy.reset(seed)

    def predict_chunk(self, rgb, depth, joint_position) -> np.ndarray:
        chunks = [
            policy.predict_chunk(rgb, depth, joint_position)
            for policy in self.policies
        ]
        return np.mean(np.stack(chunks), axis=0)

    def predict(self, rgb, depth, joint_position) -> np.ndarray:
        return self.predict_chunk(rgb, depth, joint_position)[0]


def load_policy_runner(
    checkpoint_path: Path,
    *,
    device_name: str = "auto",
) -> Any:
    """Load either the historical BC runner or a newer policy family."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    policy_type = checkpoint.get("policy_type", "behavior_cloning")
    if policy_type == "behavior_cloning":
        return BehaviorCloningRunner.from_checkpoint(
            checkpoint_path,
            device_name=device_name,
        )
    if policy_type == "diffusion":
        from buddy_manipulator.diffusion_policy import DiffusionPolicyRunner

        return DiffusionPolicyRunner.from_checkpoint(
            checkpoint_path,
            device_name=device_name,
        )
    if policy_type == "residual_diffusion":
        from buddy_manipulator.residual_diffusion import ResidualDiffusionRunner

        return ResidualDiffusionRunner.from_checkpoint(
            checkpoint_path,
            device_name=device_name,
        )
    raise ValueError(f"unsupported policy type: {policy_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a behavior-cloning policy in closed-loop MuJoCo."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--ensemble-checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Additional checkpoint to average with the primary policy; repeatable.",
    )
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
        "--diffusion-inference-steps",
        type=int,
        default=None,
        help="Override DDIM steps for diffusion checkpoints.",
    )
    parser.add_argument(
        "--diffusion-noise-scale",
        type=float,
        default=None,
        help="Scale initial diffusion noise; zero produces a deterministic mode-like sample.",
    )
    parser.add_argument(
        "--residual-blend",
        type=float,
        default=None,
        help="Override the BC-to-residual correction blend for residual diffusion.",
    )
    parser.add_argument(
        "--vision-expert-edge-margin",
        type=float,
        default=None,
        help=(
            "Route the outer workspace band of this width in meters to the "
            "calibrated RGB-D/IK grasp expert."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase4/rollout_results.json"),
    )
    return parser.parse_args()


def run_episode(
    policy: Any,
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
    if (
        args.vision_expert_edge_margin is not None
        and args.vision_expert_edge_margin <= 0
    ):
        raise ValueError("vision expert edge margin must be positive")
    policy = load_policy_runner(
        args.checkpoint,
        device_name=args.device,
    )
    if (
        args.diffusion_inference_steps is not None
        or args.diffusion_noise_scale is not None
    ):
        if not hasattr(policy, "configure_sampling"):
            raise ValueError("diffusion sampling overrides require a diffusion checkpoint")
        policy.configure_sampling(
            inference_steps=args.diffusion_inference_steps,
            initial_noise_scale=args.diffusion_noise_scale,
        )
    if args.residual_blend is not None:
        if not hasattr(policy, "residual_blend"):
            raise ValueError("--residual-blend requires a residual diffusion checkpoint")
        policy.configure_sampling(residual_blend=args.residual_blend)
    ensemble_checkpoints = [args.checkpoint, *args.ensemble_checkpoint]
    if args.ensemble_checkpoint:
        policy = AveragingPolicyRunner(
            [
                policy,
                *(
                    load_policy_runner(path, device_name=args.device)
                    for path in args.ensemble_checkpoint
                ),
            ]
        )
    if args.vision_expert_edge_margin is not None:
        policy = SpatialGatedPolicyRunner(
            primary=policy,
            specialist=ScriptedVisionGraspPolicyRunner(
                action_horizon=policy.action_horizon
            ),
            edge_margin=args.vision_expert_edge_margin,
        )
    rng = random.Random(args.seed)
    episode_results = []
    print(f"policy device: {policy.device}", flush=True)
    for episode_index in range(args.episodes):
        if hasattr(policy, "reset"):
            policy.reset(args.seed * 100_000 + episode_index)
        block_position = sample_block_position(rng)
        result = run_episode(policy, block_position, args)
        record = {
            "episode": episode_index,
            "block_position_m": list(block_position),
            **result.to_dict(),
        }
        if getattr(policy, "last_route", None) is not None:
            record["policy_route"] = policy.last_route
            if policy.detected_position is not None:
                record["detected_position_m"] = list(policy.detected_position)
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
        "policy_type": type(policy).__name__,
        "ensemble_checkpoints": [str(path) for path in ensemble_checkpoints],
        "diffusion_inference_steps": getattr(policy, "inference_steps", None),
        "diffusion_noise_scale": getattr(policy, "initial_noise_scale", None),
        "residual_blend": getattr(policy, "residual_blend", None),
        "vision_expert_edge_margin_m": args.vision_expert_edge_margin,
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
        "route_counts": {
            route: sum(
                result.get("policy_route") == route for result in episode_results
            )
            for route in ("learned_ensemble", "vision_expert")
            if any("policy_route" in result for result in episode_results)
        },
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
