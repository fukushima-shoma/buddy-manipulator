import argparse

import pytest

from buddy_manipulator.multiseed_benchmark import (
    parse_seed_list,
    summarize_benchmark,
)


def report(successes, *, x_offset=0.0):
    return {
        "episodes": [
            {
                "success": success,
                "block_position_m": [
                    0.27 + x_offset + index * 0.01,
                    0.08,
                    0.025,
                ],
            }
            for index, success in enumerate(successes)
        ]
    }


def test_parse_seed_list() -> None:
    assert parse_seed_list("7, 17,27") == [7, 17, 27]
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_seed_list("7,7")


def test_summarize_benchmark_reports_training_seed_variability(tmp_path) -> None:
    baseline_reports = {
        404: report([True, False]),
        505: report([False, True], x_offset=0.1),
    }
    candidate_reports = {
        7: {
            404: report([True, True]),
            505: report([False, True], x_offset=0.1),
        },
        17: {
            404: report([False, False]),
            505: report([True, True], x_offset=0.1),
        },
    }

    summary = summarize_benchmark(
        tmp_path / "baseline.pt",
        baseline_reports,
        {7: tmp_path / "seed7.pt", 17: tmp_path / "seed17.pt"},
        candidate_reports,
    )

    assert summary["baseline"]["successes"] == 2
    assert summary["baseline"]["success_rate"] == 0.5
    assert summary["training_seed_results"][0]["success_rate"] == 0.75
    assert summary["training_seed_results"][0]["recovered_failures"] == 1
    assert summary["training_seed_results"][1]["success_rate"] == 0.5
    assert summary["training_seed_results"][1]["recovered_failures"] == 1
    assert summary["training_seed_results"][1]["regressed_successes"] == 1
    assert summary["aggregate"]["mean_success_rate"] == 0.625
    assert summary["aggregate"]["success_rate_sample_std"] == pytest.approx(
        0.1767766953
    )


def test_summarize_benchmark_requires_candidate_seed(tmp_path) -> None:
    with pytest.raises(ValueError, match="candidate training seed"):
        summarize_benchmark(
            tmp_path / "baseline.pt",
            {404: report([True])},
            {},
            {},
        )
