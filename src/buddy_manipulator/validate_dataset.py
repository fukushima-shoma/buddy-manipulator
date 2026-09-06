"""Validate all Phase 3 demonstration files in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from buddy_manipulator.dataset import validate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate demonstration episodes.")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=Path("data/demonstrations"),
    )
    args = parser.parse_args()

    data_files = sorted(args.dataset_dir.glob("episode_*.npz"))
    if not data_files:
        raise SystemExit(f"no episodes found in {args.dataset_dir}")
    failed = 0
    for data_path in data_files:
        metadata_path = data_path.with_suffix(".json")
        if not metadata_path.exists():
            print(f"FAIL {data_path.name}: metadata file is missing")
            failed += 1
            continue
        errors = validate_episode(data_path, metadata_path)
        if errors:
            print(f"FAIL {data_path.name}: {'; '.join(errors)}")
            failed += 1
        else:
            print(f"PASS {data_path.name}")
    print(f"validated {len(data_files)} episodes; failures={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
