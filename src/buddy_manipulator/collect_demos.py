"""Collect randomized expert grasp demonstrations for imitation learning."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from buddy_manipulator.dataset import EpisodeRecorder, save_episode
from buddy_manipulator.grasp_demo import detect_block
from buddy_manipulator.grasping import execute_grasp
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, _mujoco, load_model, run_keyframes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect expert grasp episodes.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demonstrations"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    return parser.parse_args()


def next_episode_index(output_dir: Path) -> int:
    indices = []
    for path in output_dir.glob("episode_*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def set_block_position(model, data, position: tuple[float, float, float]) -> None:
    mujoco = _mujoco()
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_free"
    )
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def sample_block_position(rng: random.Random) -> tuple[float, float, float]:
    return rng.uniform(0.27, 0.33), rng.uniform(0.05, 0.11), 0.025


def collect_episode(
    output_dir: Path,
    episode_index: int,
    block_position: tuple[float, float, float],
    *,
    sample_hz: float,
    width: int,
    height: int,
    source: str = "scripted",
) -> bool:
    model, data = load_model()
    set_block_position(model, data, block_position)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
    detection = detect_block(model, data)

    with RgbdCamera(model, width=width, height=height) as camera:
        recorder = EpisodeRecorder(model, camera, sample_hz=sample_hz)
        result = execute_grasp(
            model,
            data,
            detection,
            step_callback=recorder.maybe_record,
        )
    save_episode(
        output_dir,
        episode_index,
        recorder.arrays(),
        success=result.success,
        block_start_position=block_position,
        source=source,
        termination="completed",
    )
    print(
        f"episode {episode_index:05d}: success={str(result.success).lower()} "
        f"samples={recorder.sample_count} lift={result.lift_delta:.3f}m"
    )
    return result.success


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")

    rng = random.Random(args.seed)
    start_index = next_episode_index(args.output_dir)
    successes = 0
    for offset in range(args.episodes):
        successes += collect_episode(
            args.output_dir,
            start_index + offset,
            sample_block_position(rng),
            sample_hz=args.sample_hz,
            width=args.width,
            height=args.height,
        )
    print(
        f"collection complete: {successes}/{args.episodes} successful "
        f"episodes in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
