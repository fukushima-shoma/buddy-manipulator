"""Evaluate a behavior-cloning checkpoint on recorded episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    NormalizationStats,
    choose_device,
    discover_episodes,
    load_episodes,
    select_named_episodes,
)
from buddy_manipulator.train_bc import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavior-cloning checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=Path("data/demonstrations"))
    parser.add_argument(
        "--split",
        choices=("validation", "train", "test", "all"),
        default="validation",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_dir: Path,
    *,
    split: str = "validation",
    batch_size: int = 32,
    device_name: str = "auto",
) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    normalization = NormalizationStats.from_dict(checkpoint["normalization"])
    all_paths = discover_episodes(
        dataset_dir,
        successful_only=bool(checkpoint.get("successful_only", True)),
    )
    if split == "all":
        paths = all_paths
    else:
        paths = select_named_episodes(all_paths, checkpoint[f"{split}_episodes"])
    config = checkpoint["model_config"]
    dataset = BehaviorCloningDataset(
        load_episodes(paths),
        normalization,
        action_horizon=int(config.get("action_horizon", 1)),
        phase_conditioning=int(config.get("phase_dim", 0)) > 0,
        history_horizon=int(config.get("history_horizon", 1)),
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    model = BehaviorCloningPolicy(
        image_channels=int(config["image_channels"]),
        state_dim=int(config["state_dim"]),
        action_horizon=int(config.get("action_horizon", 1)),
        use_object_features=bool(config.get("use_object_features", False)),
        use_goal_object_features=bool(
            config.get("use_goal_object_features", False)
        ),
        use_goal_target_features=bool(
            config.get("use_goal_target_features", False)
        ),
        factorized_target_heads=bool(
            config.get("factorized_target_heads", False)
        ),
        target_residual_heads=bool(
            config.get("target_residual_heads", False)
        ),
        target_residual_scale=float(
            config.get("target_residual_scale", 0.25)
        ),
        goal_dim=int(config.get("goal_dim", 0)),
        phase_dim=int(config.get("phase_dim", 0)),
        history_horizon=int(config.get("history_horizon", 1)),
        history_hidden_dim=int(config.get("history_hidden_dim", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_policy(model, loader, normalization, device)
    metrics["episodes"] = [path.name for path in paths]
    metrics["split"] = split
    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(f"split: {metrics['split']} ({len(metrics['episodes'])} episodes)")
    print(f"samples: {metrics['samples']}")
    print(f"predicted action vectors: {metrics['action_vectors']}")
    print(f"normalized MSE: {metrics['normalized_mse']:.6f}")
    print(f"mean action MAE: {metrics['mean_action_mae']:.6f}")
    print("per-actuator MAE: " + ", ".join(f"{value:.6f}" for value in metrics["action_mae"]))


if __name__ == "__main__":
    main()
