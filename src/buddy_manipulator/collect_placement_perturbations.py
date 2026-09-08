"""Collect release-pose perturbations labeled by final placement outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

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
    execute_goal_grasp,
    execute_goal_place,
    evaluate_goal_task,
    sample_object_positions,
    target_position,
)
from buddy_manipulator.grasp_success_critic import (
    GraspSuccessCriticRunner,
    OutcomeRankedGraspPosePolicy,
)
from buddy_manipulator.kinematics import JointAngles, UnreachableTargetError
from buddy_manipulator.placement_success_critic import (
    placement_context_features,
    placement_critic_features,
)
from buddy_manipulator.policy_rollout import controlled_joint_positions
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grasp_critic_checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/placement_perturbations/phase5j_seed4540.npz"),
    )
    parser.add_argument("--scenes", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=9)
    parser.add_argument("--maximum-xy-mm", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=4540)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--friction-scale-min", type=float, default=0.7)
    parser.add_argument("--friction-scale-max", type=float, default=1.3)
    parser.add_argument("--mass-scale-min", type=float, default=0.6)
    parser.add_argument("--mass-scale-max", type=float, default=1.6)
    parser.add_argument("--appearance-scale-min", type=float, default=0.7)
    parser.add_argument("--appearance-scale-max", type=float, default=1.15)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def sample_release_residuals(
    rng: np.random.Generator, count: int, maximum_xy_mm: float
) -> np.ndarray:
    if count < 2 or maximum_xy_mm <= 0:
        raise ValueError("at least two candidates and a positive range are required")
    residuals = np.zeros((count, 2), dtype=np.float32)
    residuals[1:] = rng.uniform(-maximum_xy_mm, maximum_xy_mm, (count - 1, 2))
    return residuals


def main() -> None:
    args = parse_args()
    if args.scenes <= 0:
        raise ValueError("scenes must be positive")
    grasp_policy = OutcomeRankedGraspPosePolicy(
        GraspSuccessCriticRunner.from_checkpoint(
            args.grasp_critic_checkpoint, device_name=args.device
        )
    )
    position_rng = random.Random(args.seed)
    residual_rng = np.random.default_rng(args.seed + 1)
    domain_rng = random.Random(args.seed + 1_000_003)
    randomization_config = DomainRandomizationConfig(
        finger_friction_scale=(args.friction_scale_min, args.friction_scale_max),
        object_mass_scale=(args.mass_scale_min, args.mass_scale_max),
        appearance_scale=(args.appearance_scale_min, args.appearance_scale_max),
        place_bias_jitter_mm=0.0,
    )
    features = []
    labels = []
    scene_ids = []
    residual_rows = []
    errors = []
    reachable_rows = []
    friction_rows = []
    mass_rows = []
    appearance_rows = []
    for scene_id in range(args.scenes):
        goal = ManipulationGoal(
            OBJECT_COLORS[scene_id % len(OBJECT_COLORS)],
            TARGET_COLORS[(scene_id // len(OBJECT_COLORS)) % len(TARGET_COLORS)],
        )
        positions = sample_object_positions(position_rng)
        residuals = sample_release_residuals(
            residual_rng, args.candidates, args.maximum_xy_mm
        )
        domain_sample = (
            sample_domain_randomization(domain_rng, randomization_config)
            if args.domain_randomization
            else None
        )
        successes = 0
        for residual_mm in residuals:
            model, data = load_model()
            apply_object_positions(model, data, positions)
            if domain_sample is not None:
                apply_domain_randomization(model, data, domain_sample)
            run_keyframes(
                model,
                data,
                [Keyframe(0.7, JointAngles(-1.2, 0.6, -1.2, 0.6), 0.03)],
            )
            with RgbdCamera(model, width=160, height=120) as camera:
                frame = camera.capture(data)
                detection = grasp_policy.predict_detection(frame, goal)
                execute_goal_grasp(model, data, detection)
                destination = target_position(model, data, goal)
                context = placement_context_features(
                    controlled_joint_positions(model, data), goal, destination
                )
                reachable = True
                try:
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
                except UnreachableTargetError:
                    reachable = False
                    result = evaluate_goal_task(model, data, goal)
            successes += int(result.success)
            features.append(placement_critic_features(context, residual_mm))
            labels.append(result.success)
            scene_ids.append(scene_id)
            residual_rows.append(residual_mm.copy())
            errors.append(result.placement_error_m)
            reachable_rows.append(reachable)
            friction_rows.append(
                1.0 if domain_sample is None else domain_sample.finger_friction_scale
            )
            mass_rows.append(
                1.0 if domain_sample is None else domain_sample.object_mass_scale
            )
            appearance_rows.append(
                1.0 if domain_sample is None else domain_sample.appearance_scale
            )
        print(
            f"scene {scene_id:03d}: goal={goal.object_color}->{goal.target_color} "
            f"successes={successes}/{args.candidates}",
            flush=True,
        )
    arrays = {
        "features": np.asarray(features, dtype=np.float32),
        "success": np.asarray(labels, dtype=np.float32),
        "scene_id": np.asarray(scene_ids, dtype=np.int32),
        "residual_xy_mm": np.asarray(residual_rows, dtype=np.float32),
        "placement_error_m": np.asarray(errors, dtype=np.float32),
        "reachable": np.asarray(reachable_rows, dtype=np.bool_),
        "finger_friction_scale": np.asarray(friction_rows, dtype=np.float32),
        "object_mass_scale": np.asarray(mass_rows, dtype=np.float32),
        "appearance_scale": np.asarray(appearance_rows, dtype=np.float32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    baseline = np.linalg.norm(arrays["residual_xy_mm"], axis=1) < 1e-6
    report = {
        "format_version": 1,
        "dataset": str(args.output),
        "seed": args.seed,
        "scenes": args.scenes,
        "candidates_per_scene": args.candidates,
        "samples": int(arrays["success"].size),
        "maximum_xy_mm": args.maximum_xy_mm,
        "successes": int(arrays["success"].sum()),
        "success_rate": float(arrays["success"].mean()),
        "baseline_successes": int(arrays["success"][baseline].sum()),
        "baseline_success_rate": float(arrays["success"][baseline].mean()),
        "unreachable_attempts": int((~arrays["reachable"]).sum()),
        "domain_randomization_enabled": args.domain_randomization,
        "domain_randomization_config": (
            randomization_config.to_dict() if args.domain_randomization else None
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"saved {report['samples']} attempts; success={report['success_rate']:.1%} "
        f"baseline={report['baseline_success_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
