"""Collect expert demonstrations around failed policy rollout positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

from buddy_manipulator.collect_demos import collect_episode, next_episode_index


X_BOUNDS = (0.27, 0.33)
Y_BOUNDS = (0.05, 0.11)
BLOCK_HEIGHT = 0.025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect expert corrections around failed rollout positions."
    )
    parser.add_argument("rollout_results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demonstrations"),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    return parser.parse_args()


def load_failed_positions(path: Path) -> list[tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("rollout results must contain an episodes list")
    positions = []
    for episode in episodes:
        if bool(episode.get("success", False)):
            continue
        position = episode.get("block_position_m")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("failed episode has an invalid block_position_m")
        positions.append(tuple(float(value) for value in position))
    if not positions:
        raise ValueError("rollout results contain no failed episodes")
    return positions


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(bounds[1], max(bounds[0], value))


def make_target_positions(
    failures: Sequence[tuple[float, float, float]],
    *,
    repeats: int,
    jitter: float,
    seed: int,
) -> list[tuple[float, float, float]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")
    rng = random.Random(seed)
    targets = []
    for failure in failures:
        for repeat in range(repeats):
            if repeat == 0:
                x, y = failure[:2]
            else:
                x = failure[0] + rng.uniform(-jitter, jitter)
                y = failure[1] + rng.uniform(-jitter, jitter)
            targets.append(
                (
                    _clamp(x, X_BOUNDS),
                    _clamp(y, Y_BOUNDS),
                    BLOCK_HEIGHT,
                )
            )
    return targets


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.sample_hz <= 0:
        raise ValueError("image dimensions and sample_hz must be positive")
    failures = load_failed_positions(args.rollout_results)
    targets = make_target_positions(
        failures,
        repeats=args.repeats,
        jitter=args.jitter,
        seed=args.seed,
    )
    start_index = next_episode_index(args.output_dir)
    successes = 0
    print(
        f"loaded {len(failures)} failures; collecting {len(targets)} "
        "targeted expert episodes",
        flush=True,
    )
    for offset, position in enumerate(targets):
        successes += collect_episode(
            args.output_dir,
            start_index + offset,
            position,
            sample_hz=args.sample_hz,
            width=args.width,
            height=args.height,
            source="failure_replay",
        )
    print(
        f"targeted collection complete: {successes}/{len(targets)} successful",
        flush=True,
    )


if __name__ == "__main__":
    main()
