"""Train and evaluate behavior-cloning policies across multiple random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

from buddy_manipulator.compare_rollouts import compare_reports, load_report


def parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train behavior-cloning policies with multiple seeds and evaluate "
            "them against a baseline on paired MuJoCo placements."
        )
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4/multiseed"))
    parser.add_argument("--train-seeds", type=parse_seed_list, default=[7, 17, 27])
    parser.add_argument("--rollout-seeds", type=parse_seed_list, default=[404, 505, 606])
    parser.add_argument(
        "--vary-seed",
        choices=("all", "split", "model", "sampler"),
        default="all",
        help="Training seed factor changed by --train-seeds.",
    )
    parser.add_argument(
        "--fixed-seed",
        type=int,
        default=7,
        help="Seed used for factors not selected by --vary-seed.",
    )
    parser.add_argument(
        "--baseline-reports-dir",
        type=Path,
        default=None,
        help="Optional shared directory for baseline rollout reports.",
    )
    parser.add_argument(
        "--fixed-run-dir",
        type=Path,
        default=None,
        help=(
            "Reuse this candidate directory when a varied seed equals "
            "--fixed-seed; requires --reuse-existing."
        ),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("random", "spatial"),
        default="random",
    )
    parser.add_argument("--spatial-bins", type=int, default=3)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--failure-replay-fraction", type=float, default=0.2)
    parser.add_argument(
        "--source-sampling",
        choices=("replacement", "minimal-replacement", "without-replacement"),
        default="replacement",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse checkpoints and rollout reports already present in output-dir.",
    )
    return parser.parse_args()


def _success_totals(reports: Mapping[int, dict[str, Any]]) -> tuple[int, int]:
    successes = sum(
        sum(bool(episode["success"]) for episode in report["episodes"])
        for report in reports.values()
    )
    trials = sum(len(report["episodes"]) for report in reports.values())
    if trials == 0:
        raise ValueError("rollout reports must contain at least one episode")
    return successes, trials


def resolve_factor_seeds(
    varied_seed: int,
    *,
    vary_seed: str,
    fixed_seed: int,
) -> dict[str, int]:
    if vary_seed not in {"all", "split", "model", "sampler"}:
        raise ValueError(f"unknown seed factor: {vary_seed}")
    seeds = {
        "split_seed": fixed_seed,
        "model_seed": fixed_seed,
        "sampler_seed": fixed_seed,
    }
    if vary_seed == "all":
        return {name: varied_seed for name in seeds}
    seeds[f"{vary_seed}_seed"] = varied_seed
    return seeds


def summarize_benchmark(
    baseline_checkpoint: Path,
    baseline_reports: Mapping[int, dict[str, Any]],
    candidate_checkpoints: Mapping[int, Path],
    candidate_reports: Mapping[int, Mapping[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize nested train-seed and rollout-seed paired experiments."""
    rollout_seeds = set(baseline_reports)
    if not rollout_seeds:
        raise ValueError("at least one baseline rollout report is required")
    if set(candidate_checkpoints) != set(candidate_reports):
        raise ValueError("candidate checkpoint and report training seeds must match")
    if not candidate_reports:
        raise ValueError("at least one candidate training seed is required")

    baseline_successes, baseline_trials = _success_totals(baseline_reports)
    baseline_rate = baseline_successes / baseline_trials
    training_seed_results = []
    candidate_rates = []
    deltas = []
    for train_seed in sorted(candidate_reports):
        reports = candidate_reports[train_seed]
        if set(reports) != rollout_seeds:
            raise ValueError(
                f"train seed {train_seed} does not have the baseline rollout seeds"
            )
        successes, trials = _success_totals(reports)
        if trials != baseline_trials:
            raise ValueError("baseline and candidate trial counts must match")
        recovered = 0
        regressed = 0
        rollout_results = []
        for rollout_seed in sorted(rollout_seeds):
            comparison = compare_reports(
                baseline_reports[rollout_seed],
                reports[rollout_seed],
            )
            recovered += len(comparison["recovered_episode_indices"])
            regressed += len(comparison["regressed_episode_indices"])
            rollout_results.append(
                {
                    "rollout_seed": rollout_seed,
                    "episode_count": comparison["episode_count"],
                    "baseline_successes": comparison["baseline_successes"],
                    "candidate_successes": comparison["candidate_successes"],
                    "success_rate_delta": comparison["success_rate_delta"],
                    "recovered_failures": len(
                        comparison["recovered_episode_indices"]
                    ),
                    "regressed_successes": len(
                        comparison["regressed_episode_indices"]
                    ),
                }
            )
        success_rate = successes / trials
        delta = success_rate - baseline_rate
        candidate_rates.append(success_rate)
        deltas.append(delta)
        training_seed_results.append(
            {
                "train_seed": train_seed,
                "checkpoint": str(candidate_checkpoints[train_seed]),
                "successes": successes,
                "trials": trials,
                "success_rate": success_rate,
                "success_rate_delta": delta,
                "recovered_failures": recovered,
                "regressed_successes": regressed,
                "rollouts": rollout_results,
            }
        )

    return {
        "format_version": 1,
        "baseline": {
            "checkpoint": str(baseline_checkpoint),
            "successes": baseline_successes,
            "trials": baseline_trials,
            "success_rate": baseline_rate,
        },
        "training_seed_results": training_seed_results,
        "aggregate": {
            "training_seed_count": len(candidate_rates),
            "mean_success_rate": statistics.mean(candidate_rates),
            "success_rate_sample_std": (
                statistics.stdev(candidate_rates) if len(candidate_rates) > 1 else 0.0
            ),
            "min_success_rate": min(candidate_rates),
            "max_success_rate": max(candidate_rates),
            "mean_success_rate_delta": statistics.mean(deltas),
        },
    }


def _run(command: Sequence[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _run_unless_present(
    command: Sequence[str],
    artifact: Path,
    *,
    reuse_existing: bool,
) -> None:
    if reuse_existing and artifact.exists():
        print(f"reusing: {artifact}", flush=True)
        return
    _run(command)


def _rollout_command(
    checkpoint: Path,
    output: Path,
    *,
    rollout_seed: int,
    episodes: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "buddy_manipulator.rollout_bc",
        str(checkpoint),
        "--episodes",
        str(episodes),
        "--seed",
        str(rollout_seed),
        "--headless",
        "--device",
        device,
        "--output",
        str(output),
    ]


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("episodes, epochs, and batch size must be positive")
    if args.fixed_seed < 0:
        raise ValueError("fixed seed must be non-negative")
    if args.fixed_run_dir is not None and not args.reuse_existing:
        raise ValueError("--fixed-run-dir requires --reuse-existing")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_reports = {}
    baseline_dir = args.baseline_reports_dir or args.output_dir / "baseline"
    for rollout_seed in args.rollout_seeds:
        report_path = baseline_dir / f"rollout_seed_{rollout_seed}.json"
        _run_unless_present(
            _rollout_command(
                args.baseline,
                report_path,
                rollout_seed=rollout_seed,
                episodes=args.episodes,
                device=args.device,
            ),
            report_path,
            reuse_existing=args.reuse_existing,
        )
        baseline_reports[rollout_seed] = load_report(report_path)

    candidate_checkpoints = {}
    candidate_reports = {}
    candidate_seed_configs = {}
    for train_seed in args.train_seeds:
        seed_config = resolve_factor_seeds(
            train_seed,
            vary_seed=args.vary_seed,
            fixed_seed=args.fixed_seed,
        )
        candidate_seed_configs[train_seed] = seed_config
        if train_seed == args.fixed_seed and args.fixed_run_dir is not None:
            seed_dir = args.fixed_run_dir
        else:
            seed_dir = args.output_dir / f"train_seed_{train_seed}"
        checkpoint_path = seed_dir / "bc_policy.pt"
        training_command = [
            sys.executable,
            "-m",
            "buddy_manipulator.train_bc",
            str(args.dataset_dir),
            "--output-dir",
            str(seed_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--validation-fraction",
            str(args.validation_fraction),
            "--split-strategy",
            args.split_strategy,
            "--spatial-bins",
            str(args.spatial_bins),
            "--seed",
            str(args.fixed_seed),
            "--split-seed",
            str(seed_config["split_seed"]),
            "--model-seed",
            str(seed_config["model_seed"]),
            "--sampler-seed",
            str(seed_config["sampler_seed"]),
            "--action-horizon",
            str(args.action_horizon),
            "--failure-replay-fraction",
            str(args.failure_replay_fraction),
            "--source-sampling",
            args.source_sampling,
            "--device",
            args.device,
        ]
        _run_unless_present(
            training_command,
            checkpoint_path,
            reuse_existing=args.reuse_existing,
        )
        candidate_checkpoints[train_seed] = checkpoint_path
        reports = {}
        for rollout_seed in args.rollout_seeds:
            report_path = seed_dir / f"rollout_seed_{rollout_seed}.json"
            _run_unless_present(
                _rollout_command(
                    checkpoint_path,
                    report_path,
                    rollout_seed=rollout_seed,
                    episodes=args.episodes,
                    device=args.device,
                ),
                report_path,
                reuse_existing=args.reuse_existing,
            )
            reports[rollout_seed] = load_report(report_path)
        candidate_reports[train_seed] = reports

    summary = summarize_benchmark(
        args.baseline,
        baseline_reports,
        candidate_checkpoints,
        candidate_reports,
    )
    for result in summary["training_seed_results"]:
        result["seed_config"] = candidate_seed_configs[result["train_seed"]]
    summary["config"] = {
        "dataset_dir": str(args.dataset_dir),
        "train_seeds": args.train_seeds,
        "vary_seed": args.vary_seed,
        "fixed_seed": args.fixed_seed,
        "rollout_seeds": args.rollout_seeds,
        "episodes_per_rollout_seed": args.episodes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "validation_fraction": args.validation_fraction,
        "split_strategy": args.split_strategy,
        "spatial_bins": args.spatial_bins,
        "action_horizon": args.action_horizon,
        "failure_replay_fraction": args.failure_replay_fraction,
        "source_sampling": args.source_sampling,
        "device": args.device,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    baseline = summary["baseline"]
    print(
        f"baseline: {baseline['successes']}/{baseline['trials']} "
        f"({baseline['success_rate']:.1%})",
        flush=True,
    )
    for result in summary["training_seed_results"]:
        print(
            f"{args.vary_seed} seed {result['train_seed']}: "
            f"{result['successes']}/{result['trials']} "
            f"({result['success_rate']:.1%}, "
            f"delta {result['success_rate_delta']:+.1%})",
            flush=True,
        )
    aggregate = summary["aggregate"]
    print(
        f"candidate mean: {aggregate['mean_success_rate']:.1%} +/- "
        f"{aggregate['success_rate_sample_std']:.1%} across training seeds",
        flush=True,
    )
    print(f"summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
