import pytest

from buddy_manipulator.compare_rollouts import compare_reports


def report(successes):
    return {
        "episodes": [
            {
                "success": success,
                "block_position_m": [0.27 + index * 0.01, 0.08, 0.025],
            }
            for index, success in enumerate(successes)
        ]
    }


def test_compare_reports_counts_paired_transitions() -> None:
    comparison = compare_reports(
        report([False, True, True, False]),
        report([True, False, False, False]),
    )

    assert comparison["baseline_successes"] == 2
    assert comparison["candidate_successes"] == 1
    assert comparison["recovered_episode_indices"] == [0]
    assert comparison["regressed_episode_indices"] == [1, 2]
    assert comparison["unchanged_success_indices"] == []
    assert comparison["unchanged_failure_indices"] == [3]


def test_compare_reports_rejects_different_positions() -> None:
    baseline = report([True])
    candidate = report([True])
    candidate["episodes"][0]["block_position_m"][0] = 0.31

    with pytest.raises(ValueError, match="positions do not match"):
        compare_reports(baseline, candidate)
