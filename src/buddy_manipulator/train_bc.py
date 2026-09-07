"""Train and evaluate the Phase 4 behavior-cloning baseline."""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningDataset,
    BehaviorCloningPolicy,
    NormalizationStats,
    choose_device,
    compute_normalization,
    denormalize_action,
    discover_episodes,
    load_episodes,
    select_named_episodes,
    split_episodes,
    split_episodes_spatially,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RGB-D behavior-cloning policy.")
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=Path("data/demonstrations"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase4"))
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Dataset split seed; defaults to --seed.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Model initialization seed; defaults to --seed.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="DataLoader and weighted-sampler seed; defaults to --seed.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of future actions predicted from each observation.",
    )
    parser.add_argument(
        "--use-object-features",
        action="store_true",
        help="Append a differentiable red-object RGB-D bottleneck to CNN features.",
    )
    parser.add_argument(
        "--use-goal-object-features",
        action="store_true",
        help="Append the RGB-D centroid of the object selected by the goal.",
    )
    parser.add_argument(
        "--use-goal-target-features",
        action="store_true",
        help="Append the RGB-D centroid of the target selected by the goal.",
    )
    parser.add_argument(
        "--factorized-target-heads",
        action="store_true",
        help="Use separate action decoders for green and yellow targets.",
    )
    parser.add_argument(
        "--target-residual-heads",
        action="store_true",
        help="Add small target-specific residuals to a shared action decoder.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="Initialize model weights from a compatible BC checkpoint.",
    )
    parser.add_argument(
        "--preserve-checkpoint-split",
        action="store_true",
        help="Keep checkpoint splits and add unseen episodes to training.",
    )
    parser.add_argument(
        "--reuse-checkpoint-normalization",
        action="store_true",
        help="Keep checkpoint normalization during continual fine-tuning.",
    )
    parser.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        help="Update only the action head during fine-tuning.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="Train on failed episodes too (not recommended for the baseline).",
    )
    parser.add_argument(
        "--failure-replay-fraction",
        type=float,
        default=None,
        help=(
            "Target fraction drawn from failure_replay episodes using "
            "the selected source-sampling strategy."
        ),
    )
    parser.add_argument(
        "--source-sampling",
        choices=("replacement", "minimal-replacement", "without-replacement"),
        default="replacement",
        help="How source-balanced samples are selected within each epoch.",
    )
    parser.add_argument(
        "--holdout-goal",
        default=None,
        metavar="OBJECT:TARGET",
        help="Reserve one goal combination as a test-only compositional split.",
    )
    parser.add_argument(
        "--phase-conditioning",
        action="store_true",
        help="Append the scripted task's semantic phase as a one-hot input.",
    )
    parser.add_argument(
        "--history-horizon",
        type=int,
        default=1,
        help="Number of recent proprioceptive observations encoded by a GRU.",
    )
    return parser.parse_args()


def parse_goal_pair(value: str | None) -> tuple[str, str] | None:
    if value is None:
        return None
    parts = value.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError("holdout goal must use OBJECT:TARGET")
    return parts[0], parts[1]


def failure_replay_sample_weights(
    dataset: BehaviorCloningDataset,
    failure_replay_fraction: float,
) -> torch.Tensor:
    if not 0.0 < failure_replay_fraction < 1.0:
        raise ValueError("failure_replay_fraction must be between 0 and 1")
    replay_mask = np.asarray(
        [source == "failure_replay" for source in dataset.sample_sources]
    )
    replay_count = int(replay_mask.sum())
    broad_count = len(replay_mask) - replay_count
    if replay_count == 0 or broad_count == 0:
        raise ValueError(
            "source balancing needs both failure_replay and non-replay samples"
        )
    weights = np.empty(len(replay_mask), dtype=np.float64)
    weights[replay_mask] = failure_replay_fraction / replay_count
    weights[~replay_mask] = (1.0 - failure_replay_fraction) / broad_count
    return torch.from_numpy(weights)


class SourceBalancedSampler(Sampler[int]):
    """Draw an exact source ratio without duplicate samples within an epoch."""

    def __init__(
        self,
        sample_sources: list[str],
        failure_replay_fraction: float,
        *,
        seed: int,
    ) -> None:
        if not 0.0 < failure_replay_fraction < 1.0:
            raise ValueError("failure_replay_fraction must be between 0 and 1")
        replay = [
            index
            for index, source in enumerate(sample_sources)
            if source == "failure_replay"
        ]
        broad = [
            index
            for index, source in enumerate(sample_sources)
            if source != "failure_replay"
        ]
        if not replay or not broad:
            raise ValueError(
                "source balancing needs both failure_replay and non-replay samples"
            )

        ratio = Fraction(str(failure_replay_fraction)).limit_denominator(1000)
        replay_per_unit = ratio.numerator
        broad_per_unit = ratio.denominator - ratio.numerator
        units = min(
            len(replay) // replay_per_unit,
            len(broad) // broad_per_unit,
        )
        if units <= 0:
            raise ValueError("cannot construct a source-balanced epoch")
        replay_samples = units * replay_per_unit
        broad_samples = units * broad_per_unit

        self.replay_indices = torch.tensor(replay, dtype=torch.int64)
        self.broad_indices = torch.tensor(broad, dtype=torch.int64)
        self.replay_samples = replay_samples
        self.broad_samples = broad_samples
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.replay_samples + self.broad_samples

    def __iter__(self):
        replay_order = torch.randperm(
            len(self.replay_indices), generator=self.generator
        )[: self.replay_samples]
        broad_order = torch.randperm(
            len(self.broad_indices), generator=self.generator
        )[: self.broad_samples]
        selected = torch.cat(
            [
                self.replay_indices[replay_order],
                self.broad_indices[broad_order],
            ]
        )
        order = torch.randperm(len(selected), generator=self.generator)
        return iter(selected[order].tolist())


class MinimalReplacementSourceSampler(Sampler[int]):
    """Keep epoch length and exact source ratio while minimizing duplicates."""

    def __init__(
        self,
        sample_sources: list[str],
        failure_replay_fraction: float,
        *,
        seed: int,
    ) -> None:
        if not 0.0 < failure_replay_fraction < 1.0:
            raise ValueError("failure_replay_fraction must be between 0 and 1")
        replay = [
            index
            for index, source in enumerate(sample_sources)
            if source == "failure_replay"
        ]
        broad = [
            index
            for index, source in enumerate(sample_sources)
            if source != "failure_replay"
        ]
        if not replay or not broad:
            raise ValueError(
                "source balancing needs both failure_replay and non-replay samples"
            )
        ratio = Fraction(str(failure_replay_fraction)).limit_denominator(1000)
        units = len(sample_sources) // ratio.denominator
        if units <= 0:
            raise ValueError("cannot construct a source-balanced epoch")

        self.replay_indices = torch.tensor(replay, dtype=torch.int64)
        self.broad_indices = torch.tensor(broad, dtype=torch.int64)
        self.replay_samples = units * ratio.numerator
        self.broad_samples = units * (ratio.denominator - ratio.numerator)
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.replay_samples + self.broad_samples

    def _draw(self, indices: torch.Tensor, count: int) -> torch.Tensor:
        chunks = []
        remaining = count
        while remaining > 0:
            order = torch.randperm(len(indices), generator=self.generator)
            take = min(remaining, len(indices))
            chunks.append(indices[order[:take]])
            remaining -= take
        return torch.cat(chunks)

    def __iter__(self):
        selected = torch.cat(
            [
                self._draw(self.replay_indices, self.replay_samples),
                self._draw(self.broad_indices, self.broad_samples),
            ]
        )
        order = torch.randperm(len(selected), generator=self.generator)
        return iter(selected[order].tolist())


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
            goal = batch.get("goal")
            phase = batch.get("phase")
            prediction = model(
                observation,
                joint_position,
                goal.to(device) if goal is not None else None,
                phase.to(device) if phase is not None else None,
            )
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
    failure_replay_fraction: float | None = None,
    split_seed: int | None = None,
    model_seed: int | None = None,
    sampler_seed: int | None = None,
    split_strategy: str = "random",
    spatial_bins: int = 3,
    source_sampling: str = "replacement",
    use_object_features: bool = False,
    use_goal_object_features: bool = False,
    use_goal_target_features: bool = False,
    factorized_target_heads: bool = False,
    target_residual_heads: bool = False,
    initialize_from: Path | None = None,
    preserve_checkpoint_split: bool = False,
    reuse_checkpoint_normalization: bool = False,
    freeze_image_encoder: bool = False,
    holdout_goal: tuple[str, str] | None = None,
    phase_conditioning: bool = False,
    history_horizon: int = 1,
) -> tuple[Path, dict[str, Any]]:
    if (
        epochs <= 0
        or batch_size <= 0
        or learning_rate <= 0
        or action_horizon <= 0
        or history_horizon <= 0
    ):
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    split_seed = seed if split_seed is None else split_seed
    model_seed = seed if model_seed is None else model_seed
    sampler_seed = seed if sampler_seed is None else sampler_seed
    if min(seed, split_seed, model_seed, sampler_seed) < 0:
        raise ValueError("training seeds must be non-negative")
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    paths = discover_episodes(dataset_dir, successful_only=successful_only)
    test_paths = []
    if holdout_goal is not None:
        test_paths = [path for path in paths if path.task_goal == holdout_goal]
        paths = [path for path in paths if path.task_goal != holdout_goal]
        if not test_paths:
            raise ValueError(f"holdout goal has no episodes: {holdout_goal}")
    initialization_checkpoint = None
    if initialize_from is not None:
        initialization_checkpoint = torch.load(
            initialize_from,
            map_location="cpu",
            weights_only=False,
        )
        if initialization_checkpoint.get("format_version") != 1:
            raise ValueError("unsupported initialization checkpoint format")
    if preserve_checkpoint_split:
        if initialization_checkpoint is None:
            raise ValueError("preserve_checkpoint_split requires initialize_from")
        checkpoint_train = list(initialization_checkpoint["train_episodes"])
        checkpoint_validation = list(
            initialization_checkpoint["validation_episodes"]
        )
        known_names = set(checkpoint_train) | set(checkpoint_validation)
        train_paths = select_named_episodes(paths, checkpoint_train)
        train_paths.extend(path for path in paths if path.name not in known_names)
        validation_paths = select_named_episodes(paths, checkpoint_validation)
    elif split_strategy == "random":
        train_paths, validation_paths = split_episodes(
            paths, validation_fraction=validation_fraction, seed=split_seed
        )
    elif split_strategy == "spatial":
        train_paths, validation_paths = split_episodes_spatially(
            paths,
            validation_fraction=validation_fraction,
            seed=split_seed,
            bins_per_axis=spatial_bins,
        )
    else:
        raise ValueError(f"unknown split strategy: {split_strategy}")
    train_episodes = load_episodes(train_paths)
    validation_episodes = load_episodes(validation_paths)
    test_episodes = load_episodes(test_paths)
    if reuse_checkpoint_normalization:
        if initialization_checkpoint is None:
            raise ValueError("reuse_checkpoint_normalization requires initialize_from")
        normalization = NormalizationStats.from_dict(
            initialization_checkpoint["normalization"]
        )
    else:
        normalization = compute_normalization(train_episodes)
    train_dataset = BehaviorCloningDataset(
        train_episodes,
        normalization,
        action_horizon=action_horizon,
        phase_conditioning=phase_conditioning,
        history_horizon=history_horizon,
    )
    validation_dataset = BehaviorCloningDataset(
        validation_episodes,
        normalization,
        action_horizon=action_horizon,
        phase_conditioning=phase_conditioning,
        history_horizon=history_horizon,
    )
    test_dataset = (
        BehaviorCloningDataset(
            test_episodes,
            normalization,
            action_horizon=action_horizon,
            phase_conditioning=phase_conditioning,
            history_horizon=history_horizon,
        )
        if test_episodes
        else None
    )
    if train_dataset.goal_dim != validation_dataset.goal_dim:
        raise ValueError("train and validation goal dimensions must match")
    generator = torch.Generator().manual_seed(sampler_seed)
    if failure_replay_fraction is None:
        if source_sampling != "replacement":
            raise ValueError(
                "source_sampling requires failure_replay_fraction"
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        sampling_description = "natural"
    elif source_sampling == "replacement":
        sample_weights = failure_replay_sample_weights(
            train_dataset,
            failure_replay_fraction,
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
        sampling_description = (
            f"failure_replay={failure_replay_fraction:.0%}/replacement"
        )
    elif source_sampling == "without-replacement":
        sampler = SourceBalancedSampler(
            train_dataset.sample_sources,
            failure_replay_fraction,
            seed=sampler_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
        sampling_description = (
            f"failure_replay={failure_replay_fraction:.0%}/without-replacement/"
            f"{len(sampler)} samples"
        )
    elif source_sampling == "minimal-replacement":
        sampler = MinimalReplacementSourceSampler(
            train_dataset.sample_sources,
            failure_replay_fraction,
            seed=sampler_seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
        )
        sampling_description = (
            f"failure_replay={failure_replay_fraction:.0%}/minimal-replacement/"
            f"{len(sampler)} samples"
        )
    else:
        raise ValueError(f"unknown source sampling strategy: {source_sampling}")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size)
        if test_dataset is not None
        else None
    )

    device = choose_device(device_name)
    model = BehaviorCloningPolicy(
        action_horizon=action_horizon,
        use_object_features=use_object_features,
        use_goal_object_features=use_goal_object_features,
        use_goal_target_features=use_goal_target_features,
        factorized_target_heads=factorized_target_heads,
        target_residual_heads=target_residual_heads,
        goal_dim=int(train_dataset.goal_dim or 0),
        phase_dim=train_dataset.phase_dim,
        history_horizon=history_horizon,
    ).to(device)
    if initialization_checkpoint is not None:
        initial_config = initialization_checkpoint["model_config"]
        compatible = (
            int(initial_config.get("image_channels", 4)) == 4
            and int(initial_config.get("state_dim", 6)) == 6
            and int(initial_config.get("action_dim", 6)) == 6
            and int(initial_config.get("action_horizon", 1)) == action_horizon
            and bool(initial_config.get("use_object_features", False))
            == use_object_features
            and bool(initial_config.get("use_goal_object_features", False))
            == use_goal_object_features
            and bool(initial_config.get("use_goal_target_features", False))
            == use_goal_target_features
            and bool(initial_config.get("factorized_target_heads", False))
            == factorized_target_heads
            and bool(initial_config.get("target_residual_heads", False))
            == target_residual_heads
            and int(initial_config.get("goal_dim", 0))
            == int(train_dataset.goal_dim or 0)
            and int(initial_config.get("phase_dim", 0))
            == train_dataset.phase_dim
            and int(initial_config.get("history_horizon", 1))
            == history_horizon
        )
        if not compatible:
            raise ValueError("initialization checkpoint model config is incompatible")
        model.load_state_dict(initialization_checkpoint["model_state_dict"])
    if freeze_image_encoder:
        model.image_encoder.requires_grad_(False)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    loss_function = nn.SmoothL1Loss()
    history = []
    best_state = None
    best_metrics = None
    best_epoch = 0
    best_loss = float("inf")

    if initialization_checkpoint is not None:
        initial_metrics = evaluate_policy(
            model,
            validation_loader,
            normalization,
            device,
        )
        history.append(
            {
                "epoch": 0,
                "train_loss": None,
                **initial_metrics,
            }
        )
        best_state = copy.deepcopy(model.state_dict())
        best_metrics = initial_metrics
        best_loss = initial_metrics["normalized_mse"]
        print(
            f"initial checkpoint: val_mse={best_loss:.5f} "
            f"action_mae={initial_metrics['mean_action_mae']:.5f}",
            flush=True,
        )

    print(
        f"training on {device}: {len(train_paths)} episodes/"
        f"{len(train_dataset)} samples; validation {len(validation_paths)} episodes/"
        f"{len(validation_dataset)} samples; sampling={sampling_description}; "
        f"seeds split/model/sampler={split_seed}/{model_seed}/{sampler_seed}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_image_encoder:
            model.image_encoder.eval()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation = batch["observation"].to(device)
            joint_position = batch["joint_position"].to(device)
            target = batch["action"].to(device)
            optimizer.zero_grad(set_to_none=True)
            goal = batch.get("goal")
            phase = batch.get("phase")
            prediction = model(
                observation,
                joint_position,
                goal.to(device) if goal is not None else None,
                phase.to(device) if phase is not None else None,
            )
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
    model.load_state_dict(best_state)
    test_metrics = (
        evaluate_policy(model, test_loader, normalization, device)
        if test_loader is not None
        else None
    )
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
            "use_object_features": use_object_features,
            "use_goal_object_features": use_goal_object_features,
            "use_goal_target_features": use_goal_target_features,
            "factorized_target_heads": factorized_target_heads,
            "target_residual_heads": target_residual_heads,
            "target_residual_scale": model.target_residual_scale,
            "goal_dim": int(train_dataset.goal_dim or 0),
            "phase_dim": train_dataset.phase_dim,
            "history_horizon": history_horizon,
            "history_hidden_dim": model.history_hidden_dim,
        },
        "normalization": normalization.to_dict(),
        "train_episodes": [path.name for path in train_paths],
        "validation_episodes": [path.name for path in validation_paths],
        "test_episodes": [path.name for path in test_paths],
        "holdout_goal": list(holdout_goal) if holdout_goal else None,
        "successful_only": successful_only,
        "failure_replay_fraction": failure_replay_fraction,
        "source_sampling": source_sampling,
        "split_strategy": split_strategy,
        "spatial_bins": spatial_bins,
        "seed": seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "sampler_seed": sampler_seed,
        "initialize_from": str(initialize_from) if initialize_from else None,
        "preserve_checkpoint_split": preserve_checkpoint_split,
        "reuse_checkpoint_normalization": reuse_checkpoint_normalization,
        "freeze_image_encoder": freeze_image_encoder,
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
        "use_object_features": use_object_features,
        "use_goal_object_features": use_goal_object_features,
        "use_goal_target_features": use_goal_target_features,
        "factorized_target_heads": factorized_target_heads,
        "target_residual_heads": target_residual_heads,
        "goal_dim": int(train_dataset.goal_dim or 0),
        "phase_conditioning": bool(train_dataset.phase_dim),
        "phase_dim": train_dataset.phase_dim,
        "history_horizon": history_horizon,
        "failure_replay_fraction": failure_replay_fraction,
        "source_sampling": source_sampling,
        "split_strategy": split_strategy,
        "spatial_bins": spatial_bins,
        "seed": seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "sampler_seed": sampler_seed,
        "initialize_from": str(initialize_from) if initialize_from else None,
        "preserve_checkpoint_split": preserve_checkpoint_split,
        "reuse_checkpoint_normalization": reuse_checkpoint_normalization,
        "freeze_image_encoder": freeze_image_encoder,
        "best_epoch": best_epoch,
        "train_episodes": checkpoint["train_episodes"],
        "validation_episodes": checkpoint["validation_episodes"],
        "test_episodes": checkpoint["test_episodes"],
        "holdout_goal": checkpoint["holdout_goal"],
        "best_validation_metrics": best_metrics,
        "test_metrics": test_metrics,
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
        failure_replay_fraction=args.failure_replay_fraction,
        source_sampling=args.source_sampling,
        split_strategy=args.split_strategy,
        spatial_bins=args.spatial_bins,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        sampler_seed=args.sampler_seed,
        use_object_features=args.use_object_features,
        use_goal_object_features=args.use_goal_object_features,
        use_goal_target_features=args.use_goal_target_features,
        factorized_target_heads=args.factorized_target_heads,
        target_residual_heads=args.target_residual_heads,
        initialize_from=args.initialize_from,
        preserve_checkpoint_split=args.preserve_checkpoint_split,
        reuse_checkpoint_normalization=args.reuse_checkpoint_normalization,
        freeze_image_encoder=args.freeze_image_encoder,
        holdout_goal=parse_goal_pair(args.holdout_goal),
        phase_conditioning=args.phase_conditioning,
        history_horizon=args.history_horizon,
    )


if __name__ == "__main__":
    main()
