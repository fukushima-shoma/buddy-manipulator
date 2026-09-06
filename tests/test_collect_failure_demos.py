import json

import pytest

from buddy_manipulator.collect_failure_demos import (
    X_BOUNDS,
    Y_BOUNDS,
    load_failed_positions,
    make_target_positions,
)


def test_load_failed_positions_filters_successes(tmp_path) -> None:
    path = tmp_path / "rollout.json"
    path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "success": True,
                        "block_position_m": [0.30, 0.08, 0.025],
                    },
                    {
                        "success": False,
                        "block_position_m": [0.27, 0.10, 0.025],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_failed_positions(path) == [(0.27, 0.10, 0.025)]


def test_target_positions_include_failure_and_bounded_jitter() -> None:
    targets = make_target_positions(
        [(0.27, 0.11, 0.025)],
        repeats=4,
        jitter=0.01,
        seed=4,
    )

    assert targets[0] == (0.27, 0.11, 0.025)
    assert len(targets) == 4
    assert all(X_BOUNDS[0] <= position[0] <= X_BOUNDS[1] for position in targets)
    assert all(Y_BOUNDS[0] <= position[1] <= Y_BOUNDS[1] for position in targets)
    assert all(position[2] == 0.025 for position in targets)


def test_failure_file_requires_at_least_one_failure(tmp_path) -> None:
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({"episodes": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="no failed episodes"):
        load_failed_positions(path)
