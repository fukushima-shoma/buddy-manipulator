"""Train residual diffusion on top of a frozen behavior-cloning prior."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    NormalizationStats,
    choose_device,
    compute_normalization,
    denormalize_action,
    discover_episodes,
    load_episodes,
    split_episodes,
)
from buddy_manipulator.diffusion_policy import DiffusionPolicy
from buddy_manipulator.residual_diffusion import (
    load_bc_prior,
    predict_prior_in_normalization,
)
from buddy_manipulator.train_bc import failure_replay_sample_weights
from buddy_manipulator.train_diffusion import (
    _cpu_noise,
    update_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train diffusion corrections on top of a frozen BC policy."
    )
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase4/residual_diffusion/seed7"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--failure-replay-fraction", type=float, default=0.2)
    parser.add_argument("--initial-noise-scale", type=float, default=0.0)
    parser.add_argument("--residual-blend", type=float, default=1.0)
    parser.add_argument("--train-image-encoder", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    return parser.parse_args()


@torch.no_grad()
def compute_residual_normalization(
    dataset: BehaviorCloningDataset,
    base_model: torch.nn.Module,
    base_normalization: NormalizationStats,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = []
    for batch in DataLoader(dataset, batch_size=batch_size):
        observation = batch["observation"].to(device)
        joint_position = batch["joint_position"].to(device)
        target = batch["action"].to(device)
        prior = predict_prior_in_normalization(
            base_model,
            base_normalization,
            normalization,
            observation,
            joint_position,
        )
        residuals.append((target - prior).cpu())
    values = torch.cat(residuals, dim=0)
    mean = values.mean(dim=(0, 1))
    std = values.std(dim=(0, 1), correction=0).clamp_min(0.05)
    return mean.to(device), std.to(device)


def prepare_batch(
    batch: dict[str, torch.Tensor],
    base_model: torch.nn.Module,
    base_normalization: NormalizationStats,
    normalization: NormalizationStats,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observation = batch["observation"].to(device)
    joint_position = batch["joint_position"].to(device)
    action = batch["action"].to(device)
    prior = predict_prior_in_normalization(
        base_model,
        base_normalization,
        normalization,
        observation,
        joint_position,
    )
    residual = (action - prior - residual_mean) / residual_std
    return observation, joint_position, prior, residual


def residual_noise_loss(
    model: DiffusionPolicy,
    observation: torch.Tensor,
    joint_position: torch.Tensor,
    prior: torch.Tensor,
    residual: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    timesteps = torch.randint(
        0,
        model.diffusion_steps,
        (residual.shape[0],),
        generator=generator,
        dtype=torch.long,
    ).to(residual.device)
    noise = _cpu_noise(residual.shape, generator).to(residual.device)
    noisy_residual = model.add_noise(residual, noise, timesteps)
    prediction = model(
        observation,
        joint_position,
        noisy_residual,
        timesteps,
        prior_action=prior,
    )
    return torch.nn.functional.mse_loss(prediction, noise)


@torch.no_grad()
def evaluate(
    model: DiffusionPolicy,
    loader: DataLoader,
    base_model: torch.nn.Module,
    base_normalization: NormalizationStats,
    normalization: NormalizationStats,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    device: torch.device,
    *,
    inference_steps: int,
    initial_noise_scale: float,
    residual_blend: float,
) -> dict[str, Any]:
    model.eval()
    noise_generator = torch.Generator().manual_seed(12_345)
    sample_generator = torch.Generator().manual_seed(0)
    loss_sum = 0.0
    absolute_error_sum = np.zeros(6, dtype=np.float64)
    sample_count = 0
    action_vector_count = 0
    for batch in loader:
        observation, joint_position, prior, residual = prepare_batch(
            batch,
            base_model,
            base_normalization,
            normalization,
            residual_mean,
            residual_std,
            device,
        )
        loss = residual_noise_loss(
            model,
            observation,
            joint_position,
            prior,
            residual,
            generator=noise_generator,
        )
        initial_noise = _cpu_noise(residual.shape, sample_generator).mul_(
            initial_noise_scale
        ).to(device)
        sampled_residual = model.sample(
            observation,
            joint_position,
            initial_noise,
            inference_steps=inference_steps,
            prior_action=prior,
        )
        corrected = prior + residual_blend * (
            sampled_residual * residual_std + residual_mean
        )
        target = batch["action"].to(device)
        absolute_error = torch.abs(
            denormalize_action(corrected, normalization)
            - denormalize_action(target, normalization)
        )
        absolute_error_sum += absolute_error.sum(dim=(0, 1)).cpu().numpy()
        batch_count = target.shape[0]
        loss_sum += float(loss.cpu()) * batch_count
        sample_count += batch_count
        action_vector_count += target.numel() // 6
    return {
        "noise_mse": loss_sum / sample_count,
        "samples": sample_count,
        "action_vectors": action_vector_count,
        "action_mae": (absolute_error_sum / action_vector_count).tolist(),
        "mean_action_mae": float(
            absolute_error_sum.sum() / (action_vector_count * 6)
        ),
    }


def train(
    dataset_dir: Path,
    base_checkpoint: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    validation_fraction: float = 0.2,
    seed: int = 7,
    action_horizon: int = 8,
    diffusion_steps: int = 50,
    inference_steps: int = 25,
    failure_replay_fraction: float | None = 0.2,
    initial_noise_scale: float = 0.0,
    residual_blend: float = 1.0,
    freeze_image_encoder: bool = True,
    device_name: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    if min(epochs, batch_size, action_horizon, diffusion_steps, inference_steps) <= 0:
        raise ValueError("training dimensions and durations must be positive")
    if inference_steps > diffusion_steps:
        raise ValueError("inference_steps cannot exceed diffusion_steps")
    if not 0.0 <= residual_blend <= 1.0 or initial_noise_scale < 0.0:
        raise ValueError("invalid residual sampling configuration")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device_name)
    base_model, base_normalization, base_config = load_bc_prior(
        base_checkpoint,
        device,
    )
    if base_model.action_horizon != action_horizon:
        raise ValueError("base checkpoint action horizon must match")

    paths = discover_episodes(dataset_dir, successful_only=True)
    train_paths, validation_paths = split_episodes(
        paths,
        validation_fraction=validation_fraction,
        seed=seed,
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
    residual_mean, residual_std = compute_residual_normalization(
        train_dataset,
        base_model,
        base_normalization,
        normalization,
        device,
        batch_size=batch_size,
    )

    generator = torch.Generator().manual_seed(seed)
    if failure_replay_fraction is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
    else:
        sampler = WeightedRandomSampler(
            failure_replay_sample_weights(train_dataset, failure_replay_fraction),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    model_config = {
        "image_channels": 4,
        "state_dim": 6,
        "action_dim": 6,
        "action_horizon": action_horizon,
        "diffusion_steps": diffusion_steps,
        "condition_dim": 128,
        "hidden_dim": 256,
        "residual_blocks": 3,
        "prior_action_dim": action_horizon * 6,
    }
    model = DiffusionPolicy(**model_config).to(device)
    model.image_encoder.load_state_dict(base_model.image_encoder.state_dict())
    if freeze_image_encoder:
        model.image_encoder.requires_grad_(False)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    noise_generator = torch.Generator().manual_seed(seed + 1_000_003)
    history = []
    best_state = None
    best_metrics = None
    best_epoch = 0
    best_loss = float("inf")
    update_count = 0
    print(
        f"training residual diffusion on {device}: {len(train_dataset)} train/"
        f"{len(validation_dataset)} validation samples; base={base_checkpoint}; "
        f"freeze_image_encoder={freeze_image_encoder}",
        flush=True,
    )
    print(
        "residual std: " + ", ".join(f"{value:.3f}" for value in residual_std.cpu()),
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_image_encoder:
            model.image_encoder.eval()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation, joint_position, prior, residual = prepare_batch(
                batch,
                base_model,
                base_normalization,
                normalization,
                residual_mean,
                residual_std,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = residual_noise_loss(
                model,
                observation,
                joint_position,
                prior,
                residual,
                generator=noise_generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_count += 1
            decay = min(0.995, (1.0 + update_count) / (10.0 + update_count))
            update_ema(ema_model, model, decay=decay)
            loss_sum += float(loss.detach().cpu()) * residual.shape[0]
            sample_count += residual.shape[0]
        scheduler.step()
        metrics = evaluate(
            ema_model,
            validation_loader,
            base_model,
            base_normalization,
            normalization,
            residual_mean,
            residual_std,
            device,
            inference_steps=inference_steps,
            initial_noise_scale=initial_noise_scale,
            residual_blend=residual_blend,
        )
        record = {
            "epoch": epoch,
            "train_noise_mse": loss_sum / sample_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs}: train_noise={record['train_noise_mse']:.5f} "
            f"val_noise={metrics['noise_mse']:.5f} action_mae={metrics['mean_action_mae']:.5f}",
            flush=True,
        )
        if metrics["mean_action_mae"] < best_loss:
            best_loss = metrics["mean_action_mae"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = copy.deepcopy(ema_model.state_dict())

    assert best_state is not None and best_metrics is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "residual_diffusion_policy.pt"
    checkpoint = {
        "format_version": 1,
        "policy_type": "residual_diffusion",
        "model_state_dict": best_state,
        "model_config": model_config,
        "base_model_state_dict": base_model.state_dict(),
        "base_model_config": base_config,
        "base_checkpoint": str(base_checkpoint),
        "normalization": normalization.to_dict(),
        "base_normalization": base_normalization.to_dict(),
        "residual_mean": residual_mean.cpu().tolist(),
        "residual_std": residual_std.cpu().tolist(),
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "failure_replay_fraction": failure_replay_fraction,
        "inference_steps": inference_steps,
        "inference_seed": 0,
        "initial_noise_scale": initial_noise_scale,
        "residual_blend": residual_blend,
        "best_epoch": best_epoch,
        "validation_metrics": best_metrics,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "format_version": 1,
        "status": "trained_pending_rollout",
        "experiment_name": "residual-diffusion-bc-prior",
        "hypothesis": (
            "A frozen BC visual prior prevents conditioning collapse while diffusion "
            "models corrective action residuals."
        ),
        "checkpoint": str(checkpoint_path),
        "configuration": {
            **model_config,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "failure_replay_fraction": failure_replay_fraction,
            "inference_steps": inference_steps,
            "initial_noise_scale": initial_noise_scale,
            "residual_blend": residual_blend,
            "freeze_image_encoder": freeze_image_encoder,
            "seed": seed,
        },
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "history": history,
        "decision": None,
    }
    report_path = output_dir / "experiment.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"best checkpoint: {checkpoint_path} (epoch {best_epoch})", flush=True)
    print(f"experiment record: {report_path}", flush=True)
    return checkpoint_path, report


def main() -> None:
    args = parse_args()
    train(
        args.dataset_dir,
        args.base_checkpoint,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        action_horizon=args.action_horizon,
        diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps,
        failure_replay_fraction=args.failure_replay_fraction,
        initial_noise_scale=args.initial_noise_scale,
        residual_blend=args.residual_blend,
        freeze_image_encoder=not args.train_image_encoder,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
