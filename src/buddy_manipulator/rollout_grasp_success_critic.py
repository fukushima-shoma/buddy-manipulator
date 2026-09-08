"""Evaluate outcome-ranked grasp candidates in normal and biased conditions."""

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
    detect_goal_object,
    execute_goal_grasp,
    object_position,
    sample_object_positions,
)
from buddy_manipulator.grasp_success_critic import (
    GraspSuccessCriticRunner,
    apply_grasp_residual,
    grid_candidate_residuals,
)
from buddy_manipulator.kinematics import JointAngles, UnreachableTargetError
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--controller", choices=("baseline", "critic"), default="critic")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3934)
    parser.add_argument("--known-bias-y-mm", type=float, default=0.0)
    parser.add_argument("--candidate-step-mm", type=float, default=3.0)
    parser.add_argument("--maximum-command-mm", type=float, default=9.0)
    parser.add_argument("--minimum-score-improvement", type=float, default=0.03)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/phase5/grasp_success_critic/rollout_results.json"),
    )
    return parser.parse_args()


def candidate_residuals(
    known_bias_y_mm: float,
    *,
    step_mm: float,
    maximum_command_mm: float,
) -> np.ndarray:
    return grid_candidate_residuals(
        known_bias_y_mm,
        step_mm=step_mm,
        maximum_command_mm=maximum_command_mm,
    )


def rollout(
    checkpoint: Path,
    output: Path,
    *,
    controller: str = "critic",
    episodes: int = 40,
    seed: int = 3934,
    known_bias_y_mm: float = 0.0,
    candidate_step_mm: float = 3.0,
    maximum_command_mm: float = 9.0,
    minimum_score_improvement: float = 0.03,
    width: int = 160,
    height: int = 120,
    device_name: str = "auto",
) -> dict[str, object]:
    if controller not in ("baseline", "critic"):
        raise ValueError("controller must be baseline or critic")
    if episodes <= 0 or width <= 0 or height <= 0:
        raise ValueError("episodes and image dimensions must be positive")
    runner = GraspSuccessCriticRunner.from_checkpoint(
        checkpoint, device_name=device_name
    )
    candidates = candidate_residuals(
        known_bias_y_mm,
        step_mm=candidate_step_mm,
        maximum_command_mm=maximum_command_mm,
    )
    fallback = np.asarray([0.0, known_bias_y_mm, 0.0], dtype=np.float32)
    rng = random.Random(seed)
    records = []
    for index in range(episodes):
        object_color = OBJECT_COLORS[index % len(OBJECT_COLORS)]
        distractor_color = next(color for color in OBJECT_COLORS if color != object_color)
        goal = ManipulationGoal(object_color, "green")
        positions = sample_object_positions(rng)
        model, data = load_model()
        apply_object_positions(model, data, positions)
        stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
        run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
        with RgbdCamera(model, width=width, height=height) as camera:
            frame = camera.capture(data)
            baseline = detect_goal_object(frame, goal)
            if controller == "critic":
                detection, selected_residual, probabilities = runner.choose_detection(
                    frame,
                    goal,
                    candidates,
                    fallback_residual_mm=fallback,
                    minimum_score_improvement=minimum_score_improvement,
                )
                selected_probability = float(probabilities[np.argmax(
                    np.all(np.isclose(candidates, selected_residual[None], atol=1e-5), axis=1)
                )])
            else:
                selected_residual = fallback.astype(np.float64)
                detection = apply_grasp_residual(baseline, selected_residual)
                selected_probability = None
            reachable = True
            try:
                execute_goal_grasp(model, data, detection)
            except UnreachableTargetError:
                reachable = False
                selected_residual = fallback.astype(np.float64)
                detection = apply_grasp_residual(baseline, selected_residual)
                fallback_index = int(np.flatnonzero(
                    np.all(np.isclose(candidates, fallback[None], atol=1e-5), axis=1)
                )[0])
                selected_probability = float(probabilities[fallback_index])
                execute_goal_grasp(model, data, detection)
        selected = object_position(model, data, object_color)
        distractor = object_position(model, data, distractor_color)
        success = bool(selected[2] >= 0.08 and distractor[2] <= 0.06)
        records.append(
            {
                "episode": index,
                "object_color": object_color,
                "object_start_positions_m": {
                    color: list(position) for color, position in positions.items()
                },
                "baseline_detection_m": list(baseline.position),
                "selected_residual_mm": selected_residual.tolist(),
                "selected_probability": selected_probability,
                "critic_candidate_reachable": reachable,
                "selected_final_position_m": selected.tolist(),
                "distractor_final_position_m": distractor.tolist(),
                "success": success,
            }
        )
        print(
            f"episode {index:03d}: success={str(success).lower()} object={object_color} "
            f"height={selected[2]:.3f}m residual=({selected_residual[0]:.1f},"
            f"{selected_residual[1]:.1f})mm",
            flush=True,
        )
    successes = sum(int(record["success"]) for record in records)
    result: dict[str, object] = {
        "format_version": 1,
        "controller": controller,
        "checkpoint": str(checkpoint),
        "episodes_requested": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "seed": seed,
        "known_bias_y_mm": known_bias_y_mm,
        "candidate_step_mm": candidate_step_mm,
        "maximum_command_mm": maximum_command_mm,
        "minimum_score_improvement": minimum_score_improvement,
        "stress_assumption": "the execution Y bias is known when constructing critic candidates",
        "episodes": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"grasp success: {successes}/{episodes} ({successes / episodes:.0%})")
    print(f"result JSON: {output}")
    return result


def main() -> None:
    args = parse_args()
    rollout(
        args.checkpoint,
        args.output,
        controller=args.controller,
        episodes=args.episodes,
        seed=args.seed,
        known_bias_y_mm=args.known_bias_y_mm,
        candidate_step_mm=args.candidate_step_mm,
        maximum_command_mm=args.maximum_command_mm,
        minimum_score_improvement=args.minimum_score_improvement,
        width=args.width,
        height=args.height,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
