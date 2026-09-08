"""Train a bounded grasp-pose residual on initial RGB-D observations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from buddy_manipulator.behavior_cloning import (
    EpisodePath,
    choose_device,
    discover_episodes,
    split_episodes,
)
from buddy_manipulator.goal_task import ManipulationGoal
from buddy_manipulator.grasp_pose_residual import (
    GraspPoseResidualPolicy,
    compute_feature_stats,
    grasp_pose_features,
)
from buddy_manipulator.sim_camera import RgbdFrame
from buddy_manipulator.simulation import _mujoco, load_model


def parse_goal_pair(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("goal must use OBJECT:TARGET")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir", type=Path, nargs="?", default=Path("data/goal_demonstrations")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase5/grasp_pose_residual/seed7"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--holdout-goal",
        type=parse_goal_pair,
        default=("purple", "yellow"),
    )
    parser.add_argument("--blend", type=float, default=0.25)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    return parser.parse_args()


def workspace_camera_calibration() -> tuple[float, np.ndarray, np.ndarray]:
    model, data = load_model()
    mujoco = _mujoco()
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "workspace_camera"
    )
    if camera_id < 0:
        raise KeyError("unknown camera: workspace_camera")
    return (
        float(model.cam_fovy[camera_id]),
        np.asarray(data.cam_xpos[camera_id], dtype=np.float64).copy(),
        np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy(),
    )


def load_pose_samples(
    paths: Sequence[EpisodePath],
    calibration: tuple[float, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load features, residual labels in millimeters, and baseline positions."""
    vertical_fov, camera_position, camera_rotation = calibration
    features = []
    residuals_mm = []
    baseline_positions = []
    for path in paths:
        if path.task_goal is None:
            raise ValueError(f"episode {path.name} has no manipulation goal")
        with np.load(path.data_path) as arrays:
            frame = RgbdFrame(
                rgb=arrays["rgb"][0].copy(),
                depth=arrays["depth"][0].astype(np.float32, copy=True),
                vertical_fov_degrees=vertical_fov,
                camera_position=camera_position,
                camera_rotation=camera_rotation,
            )
        goal = ManipulationGoal(*path.task_goal)
        feature, detection = grasp_pose_features(frame, goal)
        baseline = np.asarray(detection.position, dtype=np.float32)
        target_surface = np.asarray(path.block_position, dtype=np.float32)
        target_surface[2] += 0.025
        features.append(feature)
        baseline_positions.append(baseline)
        residuals_mm.append((target_surface - baseline) * 1000.0)
    return (
        np.stack(features).astype(np.float32),
        np.stack(residuals_mm).astype(np.float32),
        np.stack(baseline_positions).astype(np.float32),
    )


def pose_metrics(
    model: GraspPoseResidualPolicy,
    features: np.ndarray,
    residuals_mm: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    normalized = (features - feature_mean) / feature_std
    with torch.no_grad():
        predicted = (
            model(torch.from_numpy(normalized).to(device)).cpu().numpy()
        )
    baseline_error = np.linalg.norm(residuals_mm, axis=1)
    corrected_delta = predicted - residuals_mm
    corrected_error = np.linalg.norm(corrected_delta, axis=1)
    return {
        "samples": int(features.shape[0]),
        "baseline_mean_error_mm": float(baseline_error.mean()),
        "baseline_max_error_mm": float(baseline_error.max()),
        "corrected_mean_error_mm": float(corrected_error.mean()),
        "corrected_max_error_mm": float(corrected_error.max()),
        "corrected_axis_mae_mm": np.abs(corrected_delta).mean(axis=0).tolist(),
        "predicted_residual_mean_mm": predicted.mean(axis=0).tolist(),
    }


def train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 7,
    holdout_goal: tuple[str, str] = ("purple", "yellow"),
    blend: float = 0.25,
    device_name: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    if min(epochs, batch_size, learning_rate) <= 0:
        raise ValueError("training parameters must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = discover_episodes(dataset_dir)
    test_paths = [path for path in paths if path.task_goal == holdout_goal]
    seen_paths = [path for path in paths if path.task_goal != holdout_goal]
    if not test_paths:
        raise ValueError(f"holdout goal has no episodes: {holdout_goal}")
    train_paths, validation_paths = split_episodes(
        seen_paths, validation_fraction=validation_fraction, seed=seed
    )
    calibration = workspace_camera_calibration()
    train_features, train_residuals, _ = load_pose_samples(train_paths, calibration)
    validation_features, validation_residuals, _ = load_pose_samples(
        validation_paths, calibration
    )
    test_features, test_residuals, _ = load_pose_samples(test_paths, calibration)
    feature_stats = compute_feature_stats(train_features)
    normalized_train = (
        train_features - feature_stats.mean
    ) / feature_stats.std
    dataset = TensorDataset(
        torch.from_numpy(normalized_train),
        torch.from_numpy(train_residuals),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = choose_device(device_name)
    model = GraspPoseResidualPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics = pose_metrics(
        model,
        validation_features,
        validation_residuals,
        feature_stats.mean,
        feature_stats.std,
        device,
    )
    history = []
    print(
        f"training grasp residual on {device}: {len(train_paths)} train/"
        f"{len(validation_paths)} validation/{len(test_paths)} test episodes",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for feature_batch, residual_batch in loader:
            feature_batch = feature_batch.to(device)
            residual_batch = residual_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature_batch)
            loss = loss_function(prediction, residual_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * feature_batch.shape[0]
            sample_count += feature_batch.shape[0]
        model.eval()
        metrics = pose_metrics(
            model,
            validation_features,
            validation_residuals,
            feature_stats.mean,
            feature_stats.std,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / sample_count,
                **metrics,
            }
        )
        if metrics["corrected_mean_error_mm"] < best_metrics["corrected_mean_error_mm"]:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = metrics
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:03d}/{epochs}: train={loss_sum / sample_count:.4f} "
                f"val={metrics['corrected_mean_error_mm']:.3f}mm",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()
    test_metrics = pose_metrics(
        model,
        test_features,
        test_residuals,
        feature_stats.mean,
        feature_stats.std,
        device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "grasp_pose_residual.pt"
    checkpoint = {
        "format_version": 1,
        "policy_type": "grasp_pose_residual",
        "model_state_dict": best_state,
        "model_config": {"hidden_dim": model.hidden_dim},
        "feature_stats": feature_stats.to_dict(),
        "maximum_correction_m": [0.015, 0.015, 0.015],
        "blend": blend,
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "test_episodes": [path.name for path in test_paths],
        "holdout_goal": list(holdout_goal),
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "epochs": epochs,
        "seed": seed,
        "best_epoch": best_epoch,
        "holdout_goal": list(holdout_goal),
        "train_episodes": checkpoint["train_episodes"],
        "validation_episodes": checkpoint["validation_episodes"],
        "test_episodes": checkpoint["test_episodes"],
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best checkpoint: {checkpoint_path} (epoch {best_epoch})")
    print(
        f"held-out pose error: {test_metrics['baseline_mean_error_mm']:.3f} -> "
        f"{test_metrics['corrected_mean_error_mm']:.3f} mm"
    )
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
        holdout_goal=args.holdout_goal,
        blend=args.blend,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
