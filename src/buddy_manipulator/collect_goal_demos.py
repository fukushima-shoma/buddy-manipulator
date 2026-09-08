"""Collect goal-conditioned multi-object pick-and-place demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from buddy_manipulator.collect_demos import next_episode_index
from buddy_manipulator.dataset import EpisodeRecorder, save_episode
from buddy_manipulator.goal_task import (
    OBJECT_COLORS,
    TARGET_COLORS,
    ManipulationGoal,
    apply_object_positions,
    detect_goal_object,
    execute_pick_and_place,
    sample_object_positions,
    target_position,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect goal-conditioned pick-and-place demonstrations."
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/goal_demonstrations"),
    )
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--handoff-normalization",
        choices=("none", "canonical"),
        default="none",
    )
    return parser.parse_args()


def collect_goal_episode(
    output_dir: Path,
    episode_index: int,
    goal: ManipulationGoal,
    positions: dict[str, tuple[float, float, float]],
    *,
    sample_hz: float,
    width: int,
    height: int,
    normalize_handoff: bool = False,
) -> bool:
    model, data = load_model()
    apply_object_positions(model, data, positions)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
    with RgbdCamera(model, width=width, height=height) as camera:
        detection = detect_goal_object(camera.capture(data), goal)
        recorder = EpisodeRecorder(
            model,
            camera,
            sample_hz=sample_hz,
            object_body_name=goal.object_body_name,
        )
        result = execute_pick_and_place(
            model,
            data,
            goal,
            detection,
            normalize_handoff=normalize_handoff,
            step_callback=recorder.maybe_record,
        )
    arrays = recorder.arrays()
    arrays["goal"] = np.repeat(
        goal.vector()[None], recorder.sample_count, axis=0
    ).astype(np.float32)
    destination = target_position(model, data, goal)
    save_episode(
        output_dir,
        episode_index,
        arrays,
        success=result.success,
        block_start_position=positions[goal.object_color],
        source=(
            "goal_scripted_canonical_handoff"
            if normalize_handoff
            else "goal_scripted"
        ),
        task_goal=goal.to_dict(),
        scene_state={
            "task_version": (
                "phase5i-canonical-handoff-v1"
                if normalize_handoff
                else "phase5a-v1"
            ),
            "handoff_normalization": (
                "canonical" if normalize_handoff else "none"
            ),
            "object_start_positions_m": {
                color: list(position) for color, position in positions.items()
            },
            "target_position_m": destination.tolist(),
        },
    )
    print(
        f"episode {episode_index:05d}: success={str(result.success).lower()} "
        f"goal={goal.object_color}->{goal.target_color} "
        f"samples={recorder.sample_count} error={result.placement_error_m:.3f}m"
    )
    return result.success


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.sample_hz <= 0:
        raise ValueError("episodes and sample rate must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    goals = [
        ManipulationGoal(object_color, target_color)
        for object_color in OBJECT_COLORS
        for target_color in TARGET_COLORS
    ]
    rng = random.Random(args.seed)
    rng.shuffle(goals)
    start_index = next_episode_index(args.output_dir)
    successes = 0
    for offset in range(args.episodes):
        goal = goals[offset % len(goals)]
        successes += collect_goal_episode(
            args.output_dir,
            start_index + offset,
            goal,
            sample_object_positions(rng),
            sample_hz=args.sample_hz,
            width=args.width,
            height=args.height,
            normalize_handoff=args.handoff_normalization == "canonical",
        )
    print(
        f"goal collection complete: {successes}/{args.episodes} successful "
        f"episodes in {args.output_dir}"
    )
    if successes != args.episodes:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
