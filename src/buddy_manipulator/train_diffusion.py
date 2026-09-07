"""Train the Phase 4 conditional diffusion policy."""

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
from buddy_manipulator.train_bc import failure_replay_sample_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RGB-D conditional diffusion policy."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("data/demonstrations"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase4/diffusion"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--condition-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--failure-replay-fraction", type=float, default=0.2)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument("--experiment-name", default="diffusion-baseline")
    parser.add_argument(
        "--hypothesis",
        default=(
            "A conditional diffusion model will avoid conditional-mean action "
            "chunks and improve closed-loop grasp success over chunked BC."
        ),
    )
    return parser.parse_args()


def _cpu_noise(shape: torch.Size, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def diffusion_loss(
    model: DiffusionPolicy,
    observation: torch.Tensor,
    joint_position: torch.Tensor,
    action: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = action.shape[0]
    timesteps = torch.randint(
        0,
        model.diffusion_steps,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    ).to(action.device)
    noise = _cpu_noise(action.shape, generator).to(action.device)
    noisy_action = model.add_noise(action, noise, timesteps)
    predicted_noise = model(
        observation,
        joint_position,
        noisy_action,
        timesteps,
    )
    return torch.nn.functional.mse_loss(predicted_noise, noise)


@torch.no_grad()
def update_ema(
    ema_model: DiffusionPolicy,
    model: DiffusionPolicy,
    *,
    decay: float,
) -> None:
    for ema_value, model_value in zip(
        ema_model.state_dict().values(),
        model.state_dict().values(),
    ):
        if ema_value.is_floating_point():
            ema_value.lerp_(model_value, 1.0 - decay)
        else:
            ema_value.copy_(model_value)


@torch.no_grad()
def evaluate_denoising(
    model: DiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    seed: int = 12_345,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    weighted_loss = 0.0
    sample_count = 0
    for batch in loader:
        observation = batch["observation"].to(device)
        joint_position = batch["joint_position"].to(device)
        action = batch["action"].to(device)
        loss = diffusion_loss(
            model,
            observation,
            joint_position,
            action,
            generator=generator,
        )
        weighted_loss += float(loss.cpu()) * action.shape[0]
        sample_count += action.shape[0]
    return weighted_loss / sample_count


@torch.no_grad()
def evaluate_sampled_actions(
    model: DiffusionPolicy,
    loader: DataLoader,
    normalization: NormalizationStats,
    device: torch.device,
    *,
    inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    absolute_error_sum = np.zeros(model.action_dim, dtype=np.float64)
    element_count = 0
    sample_count = 0
    action_vector_count = 0
    for batch in loader:
        observation = batch["observation"].to(device)
        joint_position = batch["joint_position"].to(device)
        target = batch["action"].to(device)
        initial_noise = _cpu_noise(target.shape, generator).to(device)
        prediction = model.sample(
            observation,
            joint_position,
            initial_noise,
            inference_steps=inference_steps,
        )
        predicted_action = denormalize_action(prediction, normalization)
        target_action = denormalize_action(target, normalization)
        absolute_error = torch.abs(predicted_action - target_action)
        absolute_error_sum += absolute_error.sum(dim=(0, 1)).cpu().numpy()
        element_count += target.numel()
        sample_count += target.shape[0]
        action_vector_count += target.numel() // model.action_dim
    return {
        "samples": sample_count,
        "action_vectors": action_vector_count,
        "action_mae": (absolute_error_sum / action_vector_count).tolist(),
        "mean_action_mae": float(absolute_error_sum.sum() / element_count),
    }


def train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    validation_fraction: float = 0.2,
    seed: int = 7,
    split_seed: int | None = None,
    model_seed: int | None = None,
    sampler_seed: int | None = None,
    action_horizon: int = 8,
    diffusion_steps: int = 50,
    inference_steps: int = 10,
    condition_dim: int = 128,
    hidden_dim: int = 256,
    residual_blocks: int = 3,
    ema_decay: float = 0.995,
    failure_replay_fraction: float | None = 0.2,
    inference_seed: int = 0,
    device_name: str = "auto",
    experiment_name: str = "diffusion-baseline",
    hypothesis: str = (
        "A conditional diffusion model will avoid conditional-mean action chunks "
        "and improve closed-loop grasp success over chunked BC."
    ),
) -> tuple[Path, dict[str, Any]]:
    if min(epochs, batch_size, action_horizon, diffusion_steps, inference_steps) <= 0:
        raise ValueError("training dimensions and durations must be positive")
    if inference_steps > diffusion_steps:
        raise ValueError("inference_steps cannot exceed diffusion_steps")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optimizer settings are invalid")
    if not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be between zero and one")
    split_seed = seed if split_seed is None else split_seed
    model_seed = seed if model_seed is None else model_seed
    sampler_seed = seed if sampler_seed is None else sampler_seed
    if min(seed, split_seed, model_seed, sampler_seed, inference_seed) < 0:
        raise ValueError("seeds must be non-negative")

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    episode_paths = discover_episodes(dataset_dir, successful_only=True)
    train_paths, validation_paths = split_episodes(
        episode_paths,
        validation_fraction=validation_fraction,
        seed=split_seed,
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
    data_generator = torch.Generator().manual_seed(sampler_seed)
    if failure_replay_fraction is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=data_generator,
        )
        sampling_description = "natural"
    else:
        sampler = WeightedRandomSampler(
            failure_replay_sample_weights(
                train_dataset,
                failure_replay_fraction,
            ),
            num_samples=len(train_dataset),
            replacement=True,
            generator=data_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
        sampling_description = f"failure_replay={failure_replay_fraction:.0%}"
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    device = choose_device(device_name)
    model_config = {
        "image_channels": 4,
        "state_dim": 6,
        "action_dim": 6,
        "action_horizon": action_horizon,
        "diffusion_steps": diffusion_steps,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "residual_blocks": residual_blocks,
    }
    model = DiffusionPolicy(**model_config).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    noise_generator = torch.Generator().manual_seed(model_seed + 1_000_003)
    history: list[dict[str, Any]] = []
    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    update_count = 0

    print(
        f"training diffusion on {device}: {len(train_paths)} episodes/"
        f"{len(train_dataset)} samples; validation {len(validation_paths)} episodes/"
        f"{len(validation_dataset)} samples; sampling={sampling_description}; "
        f"seeds split/model/sampler={split_seed}/{model_seed}/{sampler_seed}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation = batch["observation"].to(device)
            joint_position = batch["joint_position"].to(device)
            action = batch["action"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(
                model,
                observation,
                joint_position,
                action,
                generator=noise_generator,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_count += 1
            effective_decay = min(
                ema_decay,
                (1.0 + update_count) / (10.0 + update_count),
            )
            update_ema(ema_model, model, decay=effective_decay)
            loss_sum += float(loss.detach().cpu()) * action.shape[0]
            sample_count += action.shape[0]
        scheduler.step()
        validation_loss = evaluate_denoising(
            ema_model,
            validation_loader,
            device,
        )
        record = {
            "epoch": epoch,
            "train_noise_mse": loss_sum / sample_count,
            "validation_noise_mse": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs}: train_noise={record['train_noise_mse']:.5f} "
            f"val_noise={validation_loss:.5f}",
            flush=True,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(ema_model.state_dict())

    assert best_state is not None
    ema_model.load_state_dict(best_state)
    sampled_metrics = evaluate_sampled_actions(
        ema_model,
        validation_loader,
        normalization,
        device,
        inference_steps=inference_steps,
        seed=inference_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "diffusion_policy.pt"
    checkpoint = {
        "format_version": 1,
        "policy_type": "diffusion",
        "model_state_dict": best_state,
        "model_config": model_config,
        "normalization": normalization.to_dict(),
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "successful_only": True,
        "failure_replay_fraction": failure_replay_fraction,
        "seed": seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "sampler_seed": sampler_seed,
        "inference_seed": inference_seed,
        "inference_steps": inference_steps,
        "best_epoch": best_epoch,
        "validation_noise_mse": best_validation_loss,
        "sampled_validation_metrics": sampled_metrics,
        "experiment_name": experiment_name,
        "hypothesis": hypothesis,
    }
    torch.save(checkpoint, checkpoint_path)
    report = {
        "format_version": 1,
        "status": "trained_pending_rollout",
        "experiment_name": experiment_name,
        "hypothesis": hypothesis,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "configuration": {
            **model_config,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "ema_decay": ema_decay,
            "failure_replay_fraction": failure_replay_fraction,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "sampler_seed": sampler_seed,
            "inference_steps": inference_steps,
            "inference_seed": inference_seed,
        },
        "best_epoch": best_epoch,
        "best_validation_noise_mse": best_validation_loss,
        "sampled_validation_metrics": sampled_metrics,
        "train_episodes": checkpoint["train_episodes"],
        "validation_episodes": checkpoint["validation_episodes"],
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
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        sampler_seed=args.sampler_seed,
        action_horizon=args.action_horizon,
        diffusion_steps=args.diffusion_steps,
        inference_steps=args.inference_steps,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        ema_decay=args.ema_decay,
        failure_replay_fraction=args.failure_replay_fraction,
        inference_seed=args.inference_seed,
        device_name=args.device,
        experiment_name=args.experiment_name,
        hypothesis=args.hypothesis,
    )


if __name__ == "__main__":
    main()
