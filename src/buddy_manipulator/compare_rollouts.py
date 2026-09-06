"""Compare two policy rollout reports on identical block placements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired closed-loop rollout reports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("episodes"), list):
        raise ValueError(f"report has no episodes list: {path}")
    return payload


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_episodes = baseline["episodes"]
    candidate_episodes = candidate["episodes"]
    if len(baseline_episodes) != len(candidate_episodes):
        raise ValueError("paired reports must contain the same episode count")

    recovered = []
    regressed = []
    unchanged_success = []
    unchanged_failure = []
    for index, (before, after) in enumerate(
        zip(baseline_episodes, candidate_episodes)
    ):
        if not np.allclose(
            before["block_position_m"],
            after["block_position_m"],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"episode {index} block positions do not match")
        before_success = bool(before["success"])
        after_success = bool(after["success"])
        if not before_success and after_success:
            recovered.append(index)
        elif before_success and not after_success:
            regressed.append(index)
        elif before_success:
            unchanged_success.append(index)
        else:
            unchanged_failure.append(index)

    episode_count = len(baseline_episodes)
    baseline_successes = len(regressed) + len(unchanged_success)
    candidate_successes = len(recovered) + len(unchanged_success)
    return {
        "episode_count": episode_count,
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_success_rate": baseline_successes / episode_count,
        "candidate_success_rate": candidate_successes / episode_count,
        "success_rate_delta": (
            candidate_successes - baseline_successes
        ) / episode_count,
        "recovered_episode_indices": recovered,
        "regressed_episode_indices": regressed,
        "unchanged_success_indices": unchanged_success,
        "unchanged_failure_indices": unchanged_failure,
    }


def main() -> None:
    args = parse_args()
    comparison = compare_reports(
        load_report(args.baseline),
        load_report(args.candidate),
    )
    print(
        f"baseline: {comparison['baseline_successes']}/"
        f"{comparison['episode_count']} "
        f"({comparison['baseline_success_rate']:.0%})"
    )
    print(
        f"candidate: {comparison['candidate_successes']}/"
        f"{comparison['episode_count']} "
        f"({comparison['candidate_success_rate']:.0%})"
    )
    print(f"recovered failures: {len(comparison['recovered_episode_indices'])}")
    print(f"regressed successes: {len(comparison['regressed_episode_indices'])}")
    print(f"success-rate delta: {comparison['success_rate_delta']:+.0%}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(comparison, indent=2),
            encoding="utf-8",
        )
        print(f"comparison JSON: {args.output}")


if __name__ == "__main__":
    main()
