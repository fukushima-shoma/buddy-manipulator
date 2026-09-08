"""Train an outcome critic to rank bounded grasp-pose candidates."""

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
from torch.utils.data import DataLoader, TensorDataset

from buddy_manipulator.behavior_cloning import choose_device
from buddy_manipulator.grasp_success_critic import (
    GraspSuccessCritic,
    compute_critic_feature_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/phase5/grasp_success_critic/seed17")
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def critic_metrics(
    model: GraspSuccessCritic,
    features: np.ndarray,
    labels: np.ndarray,
    scene_ids: np.ndarray,
    residuals_mm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    normalized = (features - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(normalized).to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
        bce = float(nn.functional.binary_cross_entropy_with_logits(
            logits, torch.from_numpy(labels).to(device)
        ).cpu())
    predicted = probabilities >= 0.5
    positives = labels >= 0.5
    positive_accuracy = float(predicted[positives].mean()) if positives.any() else 0.0
    negative_accuracy = float((~predicted[~positives]).mean()) if (~positives).any() else 0.0
    selected_successes = baseline_successes = oracle_successes = 0
    unique_scenes = np.unique(scene_ids)
    for scene_id in unique_scenes:
        indexes = np.flatnonzero(scene_ids == scene_id)
        selected_successes += int(labels[indexes[np.argmax(probabilities[indexes])]] >= 0.5)
        baseline_indexes = indexes[
            np.linalg.norm(residuals_mm[indexes], axis=1) < 1e-6
        ]
        if baseline_indexes.size:
            baseline_successes += int(labels[baseline_indexes[0]] >= 0.5)
        oracle_successes += int(np.any(labels[indexes] >= 0.5))
    count = max(1, len(unique_scenes))
    return {
        "samples": int(labels.size),
        "scenes": int(len(unique_scenes)),
        "bce": bce,
        "accuracy": float((predicted == positives).mean()),
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "ranking_success_rate": selected_successes / count,
        "baseline_success_rate": baseline_successes / count,
        "oracle_success_rate": oracle_successes / count,
    }


def train(
    dataset: Path,
    output_dir: Path,
    *,
    epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 17,
    device_name: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    if min(epochs, batch_size, learning_rate) <= 0:
        raise ValueError("training parameters must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with np.load(dataset) as values:
        features = values["features"].astype(np.float32)
        labels = values["success"].astype(np.float32)
        scene_ids = values["scene_id"].astype(np.int32)
        residuals_mm = values["residual_mm"].astype(np.float32)
    scenes = np.unique(scene_ids)
    if scenes.size < 2:
        raise ValueError("at least two scenes are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    validation_count = max(1, min(scenes.size - 1, round(scenes.size * validation_fraction)))
    validation_scenes = scenes[:validation_count]
    train_scenes = scenes[validation_count:]
    train_mask = np.isin(scene_ids, train_scenes)
    validation_mask = np.isin(scene_ids, validation_scenes)
    stats = compute_critic_feature_stats(features[train_mask])
    normalized_train = (features[train_mask] - stats.mean) / stats.std
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized_train), torch.from_numpy(labels[train_mask])
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = choose_device(device_name)
    model = GraspSuccessCritic().to(device)
    positives = float(labels[train_mask].sum())
    negatives = float(train_mask.sum() - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("training split must contain successful and failed grasps")
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    history = []
    print(
        f"training grasp success critic on {device}: "
        f"{len(train_scenes)} train/{len(validation_scenes)} validation scenes",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(feature_batch), label_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * feature_batch.shape[0]
            samples += feature_batch.shape[0]
        model.eval()
        metrics = critic_metrics(
            model,
            features[validation_mask],
            labels[validation_mask],
            scene_ids[validation_mask],
            residuals_mm[validation_mask],
            stats.mean,
            stats.std,
            device,
        )
        history.append({"epoch": epoch, "train_loss": loss_sum / samples, **metrics})
        score = (metrics["ranking_success_rate"], -metrics["bce"])
        best_score = (
            (-1.0, float("-inf"))
            if best_metrics is None
            else (best_metrics["ranking_success_rate"], -best_metrics["bce"])
        )
        if score > best_score:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = metrics
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:03d}/{epochs}: train={loss_sum / samples:.4f} "
                f"val_bce={metrics['bce']:.4f} rank={metrics['ranking_success_rate']:.1%}",
                flush=True,
            )
    assert best_metrics is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "grasp_success_critic.pt"
    checkpoint = {
        "format_version": 1,
        "policy_type": "grasp_success_critic",
        "model_state_dict": best_state,
        "model_config": {"hidden_dim": model.hidden_dim},
        "feature_stats": stats.to_dict(),
        "dataset": str(dataset),
        "seed": seed,
        "best_epoch": best_epoch,
        "train_scenes": train_scenes.tolist(),
        "validation_scenes": validation_scenes.tolist(),
        "validation_metrics": best_metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "feature_stats")}
    report["checkpoint"] = str(checkpoint_path)
    report["device"] = str(device)
    report["history"] = history
    (output_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"best checkpoint: {checkpoint_path} (epoch {best_epoch}); "
        f"ranking={best_metrics['ranking_success_rate']:.1%} "
        f"baseline={best_metrics['baseline_success_rate']:.1%} "
        f"oracle={best_metrics['oracle_success_rate']:.1%}"
    )
    return checkpoint_path, report


def main() -> None:
    args = parse_args()
    train(
        args.dataset,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
