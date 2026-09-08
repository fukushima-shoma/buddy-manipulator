"""Collect grasp-pose perturbations labeled by actual lift outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    ManipulationGoal,
    apply_object_positions,
    execute_goal_grasp,
    object_position,
    sample_object_positions,
)
from buddy_manipulator.grasp_pose_residual import grasp_pose_features
from buddy_manipulator.grasp_success_critic import (
    apply_grasp_residual,
    grasp_critic_features,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.kinematics import UnreachableTargetError
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/grasp_perturbations/phase5h_seed3833.npz"),
    )
    parser.add_argument("--scenes", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=9)
    parser.add_argument("--maximum-xy-mm", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=3833)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    return parser.parse_args()


def sample_residuals(
    rng: np.random.Generator,
    count: int,
    maximum_xy_mm: float,
) -> np.ndarray:
    """Return the zero baseline plus uniformly sampled XY perturbations."""
    if count < 2:
        raise ValueError("at least two candidates are required")
    if maximum_xy_mm <= 0:
        raise ValueError("maximum XY perturbation must be positive")
    residuals = np.zeros((count, 3), dtype=np.float32)
    residuals[1:, :2] = rng.uniform(
        -maximum_xy_mm, maximum_xy_mm, size=(count - 1, 2)
    )
    return residuals


def collect(
    output: Path,
    *,
    scenes: int = 60,
    candidates: int = 9,
    maximum_xy_mm: float = 9.0,
    seed: int = 3833,
    width: int = 160,
    height: int = 120,
) -> tuple[Path, dict[str, object]]:
    if scenes <= 0 or width <= 0 or height <= 0:
        raise ValueError("scenes and image dimensions must be positive")
    position_rng = random.Random(seed)
    residual_rng = np.random.default_rng(seed + 1)
    rows: dict[str, list[object]] = {
        "features": [],
        "success": [],
        "scene_id": [],
        "object_color_index": [],
        "residual_mm": [],
        "selected_final_position_m": [],
        "distractor_final_position_m": [],
        "reachable": [],
    }
    for scene_id in range(scenes):
        object_color = OBJECT_COLORS[scene_id % len(OBJECT_COLORS)]
        distractor_color = next(c for c in OBJECT_COLORS if c != object_color)
        goal = ManipulationGoal(object_color, "green")
        positions = sample_object_positions(position_rng)
        residuals = sample_residuals(residual_rng, candidates, maximum_xy_mm)
        scene_successes = 0
        for residual_mm in residuals:
            model, data = load_model()
            apply_object_positions(model, data, positions)
            stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
            run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
            with RgbdCamera(model, width=width, height=height) as camera:
                frame = camera.capture(data)
                pose_features, baseline = grasp_pose_features(frame, goal)
                detection = apply_grasp_residual(baseline, residual_mm)
                reachable = True
                try:
                    execute_goal_grasp(model, data, detection)
                except UnreachableTargetError:
                    reachable = False
            selected = object_position(model, data, object_color)
            distractor = object_position(model, data, distractor_color)
            success = bool(
                reachable and selected[2] >= 0.08 and distractor[2] <= 0.06
            )
            scene_successes += int(success)
            rows["features"].append(grasp_critic_features(pose_features, residual_mm))
            rows["success"].append(success)
            rows["scene_id"].append(scene_id)
            rows["object_color_index"].append(OBJECT_COLORS.index(object_color))
            rows["residual_mm"].append(residual_mm.copy())
            rows["selected_final_position_m"].append(selected)
            rows["distractor_final_position_m"].append(distractor)
            rows["reachable"].append(reachable)
        print(
            f"scene {scene_id:03d}: object={object_color} "
            f"successes={scene_successes}/{candidates}",
            flush=True,
        )

    arrays = {
        "features": np.asarray(rows["features"], dtype=np.float32),
        "success": np.asarray(rows["success"], dtype=np.float32),
        "scene_id": np.asarray(rows["scene_id"], dtype=np.int32),
        "object_color_index": np.asarray(rows["object_color_index"], dtype=np.int8),
        "residual_mm": np.asarray(rows["residual_mm"], dtype=np.float32),
        "selected_final_position_m": np.asarray(
            rows["selected_final_position_m"], dtype=np.float32
        ),
        "distractor_final_position_m": np.asarray(
            rows["distractor_final_position_m"], dtype=np.float32
        ),
        "reachable": np.asarray(rows["reachable"], dtype=np.bool_),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    baseline = np.linalg.norm(arrays["residual_mm"], axis=1) < 1e-6
    report: dict[str, object] = {
        "format_version": 1,
        "dataset": str(output),
        "seed": seed,
        "scenes": scenes,
        "candidates_per_scene": candidates,
        "samples": int(arrays["success"].size),
        "maximum_xy_mm": maximum_xy_mm,
        "successes": int(arrays["success"].sum()),
        "success_rate": float(arrays["success"].mean()),
        "baseline_successes": int(arrays["success"][baseline].sum()),
        "baseline_success_rate": float(arrays["success"][baseline].mean()),
        "unreachable_attempts": int((~arrays["reachable"]).sum()),
        "success_definition": "selected_z>=0.08m and distractor_z<=0.06m",
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"saved {arrays['success'].size} attempts to {output}; "
        f"success={report['success_rate']:.1%} baseline={report['baseline_success_rate']:.1%}"
    )
    return output, report


def main() -> None:
    args = parse_args()
    collect(
        args.output,
        scenes=args.scenes,
        candidates=args.candidates,
        maximum_xy_mm=args.maximum_xy_mm,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
