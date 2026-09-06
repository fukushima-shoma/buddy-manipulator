"""Train and evaluate the Phase 4 behavior-cloning baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    NormalizationStats,
    choose_device,
    compute_normalization,
    denormalize_action,
    discover_episodes,
    load_episodes,
    split_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-D behavior-cloning policy.")
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=Path("data/demonstrations"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of future actions predicted from each observation.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Train on failed episodes too (not recommended for the baseline).",
    )
    return parser.parse_args()


def evaluate_policy(
    model: nn.Module,
    loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    squared_error_sum = 0.0
    absolute_error_sum = np.zeros(6, dtype=np.float64)
    element_count = 0
    sample_count = 0
    action_vector_count = 0
    with torch.no_grad():
        for batch in loader:
            observation = batch["observation"].to(device)
            joint_position = batch["joint_position"].to(device)
            target = batch["action"].to(device)
            prediction = model(observation, joint_position)
            squared_error_sum += float(torch.square(prediction - target).sum().cpu())
            predicted_action = denormalize_action(prediction, normalization)
            target_action = denormalize_action(target, normalization)
            absolute_error = torch.abs(predicted_action - target_action)
            reduction_dimensions = tuple(range(absolute_error.ndim - 1))
            absolute_error_sum += (
                absolute_error.sum(dim=reduction_dimensions).cpu().numpy()
            )
            element_count += target.numel()
            sample_count += target.shape[0]
            action_vector_count += target.numel() // 6
    return {
        "normalized_mse": squared_error_sum / element_count,
        "action_mae": (absolute_error_sum / action_vector_count).tolist(),
        "mean_action_mae": float(absolute_error_sum.sum() / element_count),
        "samples": sample_count,
        "action_vectors": action_vector_count,
    }


def train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 7,
    device_name: str = "auto",
    successful_only: bool = True,
    action_horizon: int = 1,
) -> tuple[Path, dict[str, Any]]:
    if (
        epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0
        or action_horizon <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = discover_episodes(dataset_dir, successful_only=successful_only)
    train_paths, validation_paths = split_episodes(
        paths, validation_fraction=validation_fraction, seed=seed
    )
    train_episodes = load_episodes(train_paths)
    validation_episodes = load_episodes(validation_paths)
    normalization = compute_normalization(train_episodes)
    train_dataset = BehaviorCloningDataset(
        train_episodes,
        normalization,
        action_horizon=action_horizon,
    )
    validation_dataset = BehaviorCloningDataset(
        validation_episodes,
        normalization,
        action_horizon=action_horizon,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    device = choose_device(device_name)
    model = BehaviorCloningPolicy(action_horizon=action_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_metrics = None
    best_epoch = 0
    best_loss = float("inf")

    print(
        f"training on {device}: {len(train_paths)} episodes/"
        f"{len(train_dataset)} samples; validation {len(validation_paths)} episodes/"
        f"{len(validation_dataset)} samples",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation = batch["observation"].to(device)
            joint_position = batch["joint_position"].to(device)
            target = batch["action"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observation, joint_position)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * target.shape[0]
            sample_count += target.shape[0]
        metrics = evaluate_policy(model, validation_loader, normalization, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
            **metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs}: train={record['train_loss']:.5f} "
            f"val_mse={record['normalized_mse']:.5f} "
            f"action_mae={record['mean_action_mae']:.5f}",
            flush=True,
        )
        if record["normalized_mse"] < best_loss:
            best_loss = record["normalized_mse"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None and best_metrics is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bc_policy.pt"
    checkpoint = {
        "format_version": 1,
        "model_state_dict": best_state,
        "model_config": {
            "image_channels": 4,
            "state_dim": 6,
            "action_dim": 6,
            "action_horizon": action_horizon,
        },
        "normalization": normalization.to_dict(),
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "successful_only": successful_only,
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    metrics_path = output_dir / "training_metrics.json"
    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epochs": epochs,
        "action_horizon": action_horizon,
        "best_epoch": best_epoch,
        "train_episodes": checkpoint["train_episodes"],
        "validation_episodes": checkpoint["validation_episodes"],
        "best_validation_metrics": best_metrics,
        "history": history,
    }
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best checkpoint: {checkpoint_path} (epoch {best_epoch})", flush=True)
    print(f"metrics: {metrics_path}", flush=True)
    return checkpoint_path, report


def main() -> None:
    args = parse_args()
    train(
        args.dataset_dir,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device_name=args.device,
        successful_only=not args.include_failures,
        action_horizon=args.action_horizon,
    )


if __name__ == "__main__":
    main()
