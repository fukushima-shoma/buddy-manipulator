"""RGB-D behavior cloning data pipeline and policy network."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from buddy_manipulator.task_phase import GOAL_PHASE_DIM, encode_goal_phase


GOAL_SKILLS = ("full", "grasp", "place")
# The expert closes at 3.2 s and finishes the vertical lift at 4.7 s.  The
# overlap gives the place policy training support around an observation-driven
# handoff instead of requiring one exact transition frame.
GOAL_GRASP_END_SECONDS = 4.7
GOAL_PLACE_START_SECONDS = 3.2


@dataclass(frozen=True)
class EpisodePath:
    data_path: Path
    metadata_path: Path
    source: str
    block_position: tuple[float, float, float]
    task_goal: tuple[str, str] | None = None

    @property
    def name(self) -> str:
        return self.data_path.stem


@dataclass(frozen=True)
class EpisodeData:
    name: str
    source: str
    rgb: np.ndarray
    depth: np.ndarray
    joint_position: np.ndarray
    action: np.ndarray
    goal: np.ndarray | None = None
    timestamp: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class NormalizationStats:
    joint_mean: np.ndarray
    joint_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    depth_mean: float
    depth_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_mean": self.joint_mean.tolist(),
            "joint_std": self.joint_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "depth_mean": self.depth_mean,
            "depth_std": self.depth_std,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NormalizationStats":
        return cls(
            joint_mean=np.asarray(values["joint_mean"], dtype=np.float32),
            joint_std=np.asarray(values["joint_std"], dtype=np.float32),
            action_mean=np.asarray(values["action_mean"], dtype=np.float32),
            action_std=np.asarray(values["action_std"], dtype=np.float32),
            depth_mean=float(values["depth_mean"]),
            depth_std=float(values["depth_std"]),
        )


def discover_episodes(
    dataset_dir: Path,
    *,
    successful_only: bool = True,
) -> list[EpisodePath]:
    """Find complete episode pairs and optionally reject failed attempts."""
    episodes: list[EpisodePath] = []
    for metadata_path in sorted(dataset_dir.glob("episode_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if successful_only and not bool(metadata.get("success", False)):
            continue
        data_path = metadata_path.with_suffix(".npz")
        if not data_path.exists():
            raise FileNotFoundError(f"missing data for {metadata_path.name}")
        episodes.append(
            EpisodePath(
                data_path=data_path,
                metadata_path=metadata_path,
                source=str(metadata.get("source", "legacy")),
                block_position=tuple(
                    float(value)
                    for value in metadata["block_start_position_m"]
                ),
                task_goal=(
                    (
                        str(metadata["task_goal"]["object_color"]),
                        str(metadata["task_goal"]["target_color"]),
                    )
                    if isinstance(metadata.get("task_goal"), dict)
                    else None
                ),
            )
        )
    return episodes


def select_named_episodes(
    episodes: Sequence[EpisodePath], names: Iterable[str]
) -> list[EpisodePath]:
    by_name = {episode.name: episode for episode in episodes}
    selected = []
    for name in names:
        if name not in by_name:
            raise FileNotFoundError(f"checkpoint episode not found: {name}")
        selected.append(by_name[name])
    return selected


def split_episodes(
    episodes: Sequence[EpisodePath],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[EpisodePath], list[EpisodePath]]:
    if len(episodes) < 2:
        raise ValueError("behavior cloning needs at least two episodes")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    return shuffled[validation_count:], shuffled[:validation_count]


def split_episodes_spatially(
    episodes: Sequence[EpisodePath],
    *,
    validation_fraction: float,
    seed: int,
    bins_per_axis: int = 3,
) -> tuple[list[EpisodePath], list[EpisodePath]]:
    """Split episodes while preserving spatial and collection-source coverage."""
    if len(episodes) < 2:
        raise ValueError("behavior cloning needs at least two episodes")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if bins_per_axis <= 0:
        raise ValueError("bins_per_axis must be positive")

    x_values = [episode.block_position[0] for episode in episodes]
    y_values = [episode.block_position[1] for episode in episodes]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    def bin_index(value: float, lower: float, upper: float) -> int:
        if upper <= lower:
            return 0
        normalized = (value - lower) / (upper - lower)
        return min(bins_per_axis - 1, max(0, int(normalized * bins_per_axis)))

    strata: dict[tuple[int, int, str], list[EpisodePath]] = {}
    for episode in episodes:
        key = (
            bin_index(episode.block_position[0], x_min, x_max),
            bin_index(episode.block_position[1], y_min, y_max),
            episode.source,
        )
        strata.setdefault(key, []).append(episode)

    validation_count = min(
        len(episodes) - 1,
        max(1, round(len(episodes) * validation_fraction)),
    )
    capacities = {key: max(0, len(group) - 1) for key, group in strata.items()}
    if sum(capacities.values()) < validation_count:
        raise ValueError("cannot retain one training episode per spatial/source stratum")
    allocations = {
        key: min(capacities[key], int(len(group) * validation_fraction))
        for key, group in strata.items()
    }
    rng = random.Random(seed)
    tie_breakers = {key: rng.random() for key in strata}
    while sum(allocations.values()) < validation_count:
        eligible = [
            key for key in strata if allocations[key] < capacities[key]
        ]
        selected = max(
            eligible,
            key=lambda key: (
                len(strata[key]) * validation_fraction - allocations[key],
                tie_breakers[key],
            ),
        )
        allocations[selected] += 1

    train_episodes = []
    validation_episodes = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda episode: episode.name)
        rng.shuffle(group)
        split_index = allocations[key]
        validation_episodes.extend(group[:split_index])
        train_episodes.extend(group[split_index:])
    return (
        sorted(train_episodes, key=lambda episode: episode.name),
        sorted(validation_episodes, key=lambda episode: episode.name),
    )


def load_episodes(episode_paths: Sequence[EpisodePath]) -> list[EpisodeData]:
    episodes = []
    for episode_path in episode_paths:
        with np.load(episode_path.data_path) as arrays:
            episodes.append(
                EpisodeData(
                    name=episode_path.name,
                    source=episode_path.source,
                    rgb=arrays["rgb"].copy(),
                    depth=arrays["depth"].astype(np.float32, copy=True),
                    joint_position=arrays["joint_position"].astype(
                        np.float32, copy=True
                    ),
                    action=arrays["action"].astype(np.float32, copy=True),
                    goal=(
                        arrays["goal"].astype(np.float32, copy=True)
                        if "goal" in arrays.files
                        else None
                    ),
                    timestamp=arrays["timestamp"].astype(np.float64, copy=True),
                )
            )
    return episodes


def slice_goal_skill_episodes(
    episodes: Sequence[EpisodeData],
    skill: str,
) -> list[EpisodeData]:
    """Return episode views for one hierarchical goal-manipulation skill."""
    if skill not in GOAL_SKILLS:
        raise ValueError(f"unsupported goal skill: {skill}")
    if skill == "full":
        return list(episodes)

    sliced = []
    for episode in episodes:
        if episode.timestamp is None:
            raise ValueError("goal skill slicing requires episode timestamps")
        elapsed = episode.timestamp - episode.timestamp[0]
        if skill == "grasp":
            indices = np.flatnonzero(elapsed <= GOAL_GRASP_END_SECONDS + 1e-6)
        else:
            indices = np.flatnonzero(elapsed >= GOAL_PLACE_START_SECONDS - 1e-6)
        if indices.size == 0:
            raise ValueError(f"episode {episode.name} has no {skill} samples")
        sliced_goal = (
            episode.goal[indices].copy() if episode.goal is not None else None
        )
        if skill == "grasp" and sliced_goal is not None:
            # Grasp acquisition is shared across destinations. Removing target
            # bits prevents the held-out object-target pair becoming a shortcut.
            sliced_goal[:, 2:4] = 0.0
        elif skill == "place" and sliced_goal is not None:
            # Transport is shared across object colors once the object is held.
            # Keep only the destination to expose the intended composition.
            sliced_goal[:, 0:2] = 0.0
        sliced.append(
            EpisodeData(
                name=episode.name,
                source=episode.source,
                rgb=episode.rgb[indices],
                depth=episode.depth[indices],
                joint_position=episode.joint_position[indices],
                action=episode.action[indices],
                goal=sliced_goal,
                timestamp=episode.timestamp[indices],
            )
        )
    return sliced


def _safe_std(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    std = np.asarray(values.std(axis=axis), dtype=np.float32)
    return np.maximum(std, np.float32(1e-6))


def compute_normalization(episodes: Sequence[EpisodeData]) -> NormalizationStats:
    if not episodes:
        raise ValueError("cannot normalize an empty dataset")
    joints = np.concatenate([episode.joint_position for episode in episodes])
    actions = np.concatenate([episode.action for episode in episodes])
    depth_count = sum(episode.depth.size for episode in episodes)
    depth_sum = sum(float(episode.depth.sum(dtype=np.float64)) for episode in episodes)
    depth_squared_sum = sum(
        float(np.square(episode.depth, dtype=np.float64).sum())
        for episode in episodes
    )
    depth_mean = depth_sum / depth_count
    depth_variance = max(depth_squared_sum / depth_count - depth_mean**2, 1e-12)
    return NormalizationStats(
        joint_mean=joints.mean(axis=0).astype(np.float32),
        joint_std=_safe_std(joints, axis=0),
        action_mean=actions.mean(axis=0).astype(np.float32),
        action_std=_safe_std(actions, axis=0),
        depth_mean=depth_mean,
        depth_std=max(depth_variance**0.5, 1e-6),
    )


class BehaviorCloningDataset(Dataset):
    """Frame-level samples, split at the episode level before construction."""

    def __init__(
        self,
        episodes: Sequence[EpisodeData],
        normalization: NormalizationStats,
        *,
        action_horizon: int = 1,
        phase_conditioning: bool = False,
        history_horizon: int = 1,
    ) -> None:
        if action_horizon <= 0 or history_horizon <= 0:
            raise ValueError("action and history horizons must be positive")
        self.episodes = list(episodes)
        self.normalization = normalization
        self.action_horizon = action_horizon
        self.phase_dim = GOAL_PHASE_DIM if phase_conditioning else 0
        self.history_horizon = history_horizon
        goal_dimensions = {
            None if episode.goal is None else int(episode.goal.shape[1])
            for episode in self.episodes
        }
        if len(goal_dimensions) != 1:
            raise ValueError("episodes must consistently include goal vectors")
        self.goal_dim = next(iter(goal_dimensions))
        self._sample_index = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(self.episodes)
            for frame_index in range(episode.sample_count)
        ]
        self.sample_sources = [
            episode.source
            for episode in self.episodes
            for _ in range(episode.sample_count)
        ]

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame_index = self._sample_index[index]
        episode = self.episodes[episode_index]
        observation, joint = normalize_observation(
            episode.rgb[frame_index],
            episode.depth[frame_index],
            episode.joint_position[frame_index],
            self.normalization,
        )
        action_end = min(
            episode.sample_count,
            frame_index + self.action_horizon,
        )
        actions = episode.action[frame_index:action_end]
        if actions.shape[0] < self.action_horizon:
            padding = np.repeat(
                actions[-1:],
                self.action_horizon - actions.shape[0],
                axis=0,
            )
            actions = np.concatenate([actions, padding], axis=0)
        action = (
            actions - self.normalization.action_mean
        ) / self.normalization.action_std
        if self.action_horizon == 1:
            action = action[0]
        sample = {
            "observation": torch.from_numpy(observation),
            "action": torch.from_numpy(action.astype(np.float32)),
        }
        if self.history_horizon == 1:
            sample["joint_position"] = torch.from_numpy(joint.astype(np.float32))
        else:
            history_indices = [
                max(0, frame_index - self.history_horizon + 1 + offset)
                for offset in range(self.history_horizon)
            ]
            joint_history = episode.joint_position[history_indices]
            normalized_history = (
                joint_history - self.normalization.joint_mean
            ) / self.normalization.joint_std
            sample["joint_position"] = torch.from_numpy(
                normalized_history.astype(np.float32)
            )
        if episode.goal is not None:
            sample["goal"] = torch.from_numpy(
                episode.goal[frame_index].astype(np.float32)
            )
        if self.phase_dim:
            if episode.timestamp is None:
                raise ValueError("phase conditioning requires episode timestamps")
            elapsed_seconds = float(
                episode.timestamp[frame_index] - episode.timestamp[0]
            )
            sample["phase"] = torch.from_numpy(encode_goal_phase(elapsed_seconds))
        return sample


def normalize_observation(
    rgb: np.ndarray,
    depth: np.ndarray,
    joint_position: np.ndarray,
    normalization: NormalizationStats,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth shape must match rgb spatial dimensions")
    if joint_position.shape != (6,):
        raise ValueError("joint_position must have shape (6,)")
    normalized_rgb = rgb.astype(np.float32) / 255.0
    normalized_depth = (
        depth.astype(np.float32) - normalization.depth_mean
    ) / normalization.depth_std
    normalized_depth = np.clip(normalized_depth, -5.0, 5.0)
    observation = np.concatenate(
        [normalized_rgb, normalized_depth[..., None]], axis=-1
    )
    observation = np.ascontiguousarray(observation.transpose(2, 0, 1))
    normalized_joint = (
        joint_position.astype(np.float32) - normalization.joint_mean
    ) / normalization.joint_std
    return observation, normalized_joint.astype(np.float32)


class MpsSafeAdaptiveAvgPool2d(nn.Module):
    """Adaptive 2D average pooling using MPS-supported tensor reductions."""

    def __init__(self, output_size: tuple[int, int]) -> None:
        super().__init__()
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output dimensions must be positive")
        self.output_size = output_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        height, width = values.shape[-2:]
        output_height, output_width = self.output_size
        rows = []
        for row in range(output_height):
            row_start = row * height // output_height
            row_end = ((row + 1) * height + output_height - 1) // output_height
            columns = []
            for column in range(output_width):
                column_start = column * width // output_width
                column_end = (
                    (column + 1) * width + output_width - 1
                ) // output_width
                columns.append(
                    values[
                        ...,
                        row_start:row_end,
                        column_start:column_end,
                    ].mean(dim=(-2, -1))
                )
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)


def extract_red_object_features(observation: torch.Tensor) -> torch.Tensor:
    """Extract a compact image-space object location from normalized RGB-D."""
    if observation.ndim != 4 or observation.shape[1] != 4:
        raise ValueError("observation must have shape (batch, 4, height, width)")
    red, green, blue, depth = observation.unbind(dim=1)
    mask = (
        (red > 0.35)
        & (red > green * 1.35)
        & (red > blue * 1.35)
    ).to(dtype=observation.dtype)
    height, width = mask.shape[-2:]
    x_coordinates = torch.linspace(
        -1.0,
        1.0,
        width,
        dtype=observation.dtype,
        device=observation.device,
    ).reshape(1, 1, width)
    y_coordinates = torch.linspace(
        -1.0,
        1.0,
        height,
        dtype=observation.dtype,
        device=observation.device,
    ).reshape(1, height, 1)
    pixel_count = mask.sum(dim=(1, 2))
    safe_pixel_count = pixel_count.clamp_min(1.0)
    centroid_x = (mask * x_coordinates).sum(dim=(1, 2)) / safe_pixel_count
    centroid_y = (mask * y_coordinates).sum(dim=(1, 2)) / safe_pixel_count
    mean_depth = (mask * depth).sum(dim=(1, 2)) / safe_pixel_count
    area_percent = pixel_count * (100.0 / float(height * width))
    return torch.stack(
        [centroid_x, centroid_y, mean_depth, area_percent],
        dim=1,
    )


def extract_goal_object_features(
    observation: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    """Extract the image-space location of the object selected by the goal."""
    if observation.ndim != 4 or observation.shape[1] != 4:
        raise ValueError("observation must have shape (batch, 4, height, width)")
    if goal.ndim != 2 or goal.shape[0] != observation.shape[0] or goal.shape[1] < 2:
        raise ValueError("goal must select red or purple for every observation")
    red, green, blue, depth = observation.unbind(dim=1)
    masks = (
        (red > 0.35) & (red > green * 1.35) & (red > blue * 1.35),
        (red > 0.35)
        & (blue > 0.35)
        & (red > green * 1.35)
        & (blue > green * 1.35),
    )
    height, width = red.shape[-2:]
    x_coordinates = torch.linspace(
        -1.0, 1.0, width, dtype=observation.dtype, device=observation.device
    ).reshape(1, 1, width)
    y_coordinates = torch.linspace(
        -1.0, 1.0, height, dtype=observation.dtype, device=observation.device
    ).reshape(1, height, 1)
    color_features = []
    for raw_mask in masks:
        mask = raw_mask.to(dtype=observation.dtype)
        pixel_count = mask.sum(dim=(1, 2))
        safe_pixel_count = pixel_count.clamp_min(1.0)
        color_features.append(
            torch.stack(
                [
                    (mask * x_coordinates).sum(dim=(1, 2)) / safe_pixel_count,
                    (mask * y_coordinates).sum(dim=(1, 2)) / safe_pixel_count,
                    (mask * depth).sum(dim=(1, 2)) / safe_pixel_count,
                    pixel_count * (100.0 / float(height * width)),
                ],
                dim=1,
            )
        )
    return (
        color_features[0] * goal[:, 0:1]
        + color_features[1] * goal[:, 1:2]
    )


def extract_goal_target_features(
    observation: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    """Extract the image-space location of the target selected by the goal."""
    if observation.ndim != 4 or observation.shape[1] != 4:
        raise ValueError("observation must have shape (batch, 4, height, width)")
    if goal.ndim != 2 or goal.shape[0] != observation.shape[0] or goal.shape[1] < 4:
        raise ValueError("goal must select green or yellow for every observation")
    red, green, blue, depth = observation.unbind(dim=1)
    height, width = red.shape[-2:]
    x_coordinates = torch.linspace(
        -1.0, 1.0, width, dtype=observation.dtype, device=observation.device
    ).reshape(1, 1, width)
    y_coordinates = torch.linspace(
        -1.0, 1.0, height, dtype=observation.dtype, device=observation.device
    ).reshape(1, height, 1)
    target_region = (x_coordinates > -0.2) & (y_coordinates > 0.35)
    masks = (
        (green > 0.35)
        & (blue > 0.30)
        & (green > red * 1.35)
        & (blue > red * 1.20)
        & (blue < green * 1.02)
        & target_region,
        (red > 0.45)
        & (green > 0.40)
        & (red > blue * 1.50)
        & (green > blue * 1.50)
        & (green > red * 0.80)
        & (red > green * 0.70)
        & target_region,
    )
    target_features = []
    for raw_mask in masks:
        mask = raw_mask.to(dtype=observation.dtype)
        pixel_count = mask.sum(dim=(1, 2))
        safe_pixel_count = pixel_count.clamp_min(1.0)
        target_features.append(
            torch.stack(
                [
                    (mask * x_coordinates).sum(dim=(1, 2)) / safe_pixel_count,
                    (mask * y_coordinates).sum(dim=(1, 2)) / safe_pixel_count,
                    (mask * depth).sum(dim=(1, 2)) / safe_pixel_count,
                    pixel_count * (100.0 / float(height * width)),
                ],
                dim=1,
            )
        )
    return (
        target_features[0] * goal[:, 2:3]
        + target_features[1] * goal[:, 3:4]
    )


class BehaviorCloningPolicy(nn.Module):
    """Small CNN policy for RGB-D and proprioceptive observations."""

    def __init__(
        self,
        *,
        image_channels: int = 4,
        state_dim: int = 6,
        action_horizon: int = 1,
        use_object_features: bool = False,
        use_goal_object_features: bool = False,
        use_goal_target_features: bool = False,
        factorized_target_heads: bool = False,
        target_residual_heads: bool = False,
        target_residual_scale: float = 0.25,
        goal_dim: int = 0,
        phase_dim: int = 0,
        history_horizon: int = 1,
        history_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if action_horizon <= 0 or history_horizon <= 0 or history_hidden_dim <= 0:
            raise ValueError("action, history, and hidden dimensions must be positive")
        if use_goal_object_features and goal_dim < 2:
            raise ValueError("goal object features require an object-conditioned goal")
        if use_goal_target_features and goal_dim < 4:
            raise ValueError("goal target features require a target-conditioned goal")
        if factorized_target_heads and goal_dim < 4:
            raise ValueError("factorized target heads require a target-conditioned goal")
        if target_residual_heads and goal_dim < 4:
            raise ValueError("target residual heads require a target-conditioned goal")
        if factorized_target_heads and target_residual_heads:
            raise ValueError("target decoder modes are mutually exclusive")
        if target_residual_scale <= 0:
            raise ValueError("target_residual_scale must be positive")
        self.action_horizon = action_horizon
        self.use_object_features = use_object_features
        self.use_goal_object_features = use_goal_object_features
        self.use_goal_target_features = use_goal_target_features
        self.factorized_target_heads = factorized_target_heads
        self.target_residual_heads = target_residual_heads
        self.target_residual_scale = target_residual_scale
        self.goal_dim = goal_dim
        self.phase_dim = phase_dim
        self.state_dim = state_dim
        self.history_horizon = history_horizon
        self.history_hidden_dim = history_hidden_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            MpsSafeAdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.history_encoder = (
            nn.GRU(
                input_size=state_dim,
                hidden_size=history_hidden_dim,
                batch_first=True,
            )
            if history_horizon > 1
            else None
        )
        state_feature_dim = (
            history_hidden_dim if history_horizon > 1 else state_dim
        )
        action_input_dim = (
            64 * 2 * 2
            + state_feature_dim
            + (4 if use_object_features else 0)
            + (4 if use_goal_object_features else 0)
            + (4 if use_goal_target_features else 0)
            + goal_dim
            + phase_dim
        )

        def build_action_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(action_input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 6 * action_horizon),
            )

        self.action_head = (
            nn.ModuleList([build_action_head(), build_action_head()])
            if factorized_target_heads
            else build_action_head()
        )
        self.target_residual_head = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(action_input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 6 * action_horizon),
                    )
                    for _ in range(2)
                ]
            )
            if target_residual_heads
            else None
        )
        if self.target_residual_head is not None:
            for head in self.target_residual_head:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        observation: torch.Tensor,
        joint_position: torch.Tensor,
        goal: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_features = self.image_encoder(observation)
        if self.history_encoder is None:
            if joint_position.shape != (observation.shape[0], self.state_dim):
                raise ValueError(
                    f"joint_position must have shape (batch, {self.state_dim})"
                )
            state_features = joint_position
        else:
            expected_shape = (
                observation.shape[0],
                self.history_horizon,
                self.state_dim,
            )
            if joint_position.shape != expected_shape:
                raise ValueError(
                    "joint_position must have shape "
                    f"(batch, {self.history_horizon}, {self.state_dim})"
                )
            _, hidden = self.history_encoder(joint_position)
            state_features = hidden[-1]
        policy_features = [image_features, state_features]
        if self.use_object_features:
            policy_features.append(extract_red_object_features(observation))
        if self.use_goal_object_features:
            if goal is None:
                raise ValueError("goal object features require a goal")
            policy_features.append(extract_goal_object_features(observation, goal))
        if self.use_goal_target_features:
            if goal is None:
                raise ValueError("goal target features require a goal")
            policy_features.append(extract_goal_target_features(observation, goal))
        if self.goal_dim:
            if goal is None or goal.shape != (observation.shape[0], self.goal_dim):
                raise ValueError(
                    f"goal must have shape (batch, {self.goal_dim})"
                )
            policy_features.append(goal)
        elif goal is not None:
            raise ValueError("this policy checkpoint is not goal-conditioned")
        if self.phase_dim:
            if phase is None or phase.shape != (observation.shape[0], self.phase_dim):
                raise ValueError(
                    f"phase must have shape (batch, {self.phase_dim})"
                )
            policy_features.append(phase)
        elif phase is not None:
            raise ValueError("this policy checkpoint is not phase-conditioned")
        features = torch.cat(policy_features, dim=1)
        if self.factorized_target_heads:
            assert goal is not None
            head_actions = torch.stack(
                [head(features) for head in self.action_head],
                dim=1,
            )
            action = (
                head_actions * goal[:, 2:4].unsqueeze(-1)
            ).sum(dim=1)
        else:
            action = self.action_head(features)
            if self.target_residual_head is not None:
                assert goal is not None
                residuals = torch.stack(
                    [head(features) for head in self.target_residual_head],
                    dim=1,
                )
                selected_residual = (
                    residuals * goal[:, 2:4].unsqueeze(-1)
                ).sum(dim=1)
                action = action + self.target_residual_scale * selected_residual
        if self.action_horizon == 1:
            return action
        return action.reshape(action.shape[0], self.action_horizon, 6)


def denormalize_action(
    normalized_action: torch.Tensor,
    normalization: NormalizationStats,
) -> torch.Tensor:
    mean = torch.as_tensor(
        normalization.action_mean,
        device=normalized_action.device,
        dtype=normalized_action.dtype,
    )
    std = torch.as_tensor(
        normalization.action_std,
        device=normalized_action.device,
        dtype=normalized_action.dtype,
    )
    return normalized_action * std + mean


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class BehaviorCloningRunner:
    model: BehaviorCloningPolicy
    normalization: NormalizationStats
    device: torch.device
    goal_skill: str = "full"
    _joint_history: list[np.ndarray] = field(default_factory=list, init=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
    ) -> "BehaviorCloningRunner":
        device = choose_device(device_name)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        config = checkpoint["model_config"]
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
        model.eval()
        return cls(
            model=model,
            normalization=NormalizationStats.from_dict(
                checkpoint["normalization"]
            ),
            device=device,
            goal_skill=str(checkpoint.get("goal_skill", "full")),
        )

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_position: np.ndarray,
    ) -> np.ndarray:
        return self.predict_chunk(rgb, depth, joint_position)[0]

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._joint_history.clear()

    def prime_joint_history(self, joint_positions: Sequence[np.ndarray]) -> None:
        """Seed recurrent proprioception from observations made by another skill."""
        if self.model.history_horizon == 1:
            return
        for joint_position in joint_positions[-self.model.history_horizon :]:
            values = np.asarray(joint_position, dtype=np.float32)
            if values.shape != (6,):
                raise ValueError("joint history values must have shape (6,)")
            normalized = (
                values - self.normalization.joint_mean
            ) / self.normalization.joint_std
            self._joint_history.append(normalized.astype(np.float32))

    def predict_chunk(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_position: np.ndarray,
        goal: np.ndarray | None = None,
        phase: np.ndarray | None = None,
    ) -> np.ndarray:
        observation, normalized_joint = normalize_observation(
            rgb,
            depth,
            joint_position,
            self.normalization,
        )
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).to(
            self.device
        )
        if self.model.history_horizon == 1:
            joint_input = normalized_joint
        else:
            self._joint_history.append(normalized_joint.copy())
            self._joint_history = self._joint_history[-self.model.history_horizon :]
            padded_history = [self._joint_history[0]] * (
                self.model.history_horizon - len(self._joint_history)
            )
            joint_input = np.stack([*padded_history, *self._joint_history])
        joint_tensor = torch.from_numpy(joint_input).unsqueeze(0).to(self.device)
        goal_tensor = (
            torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(self.device)
            if goal is not None
            else None
        )
        phase_tensor = (
            torch.from_numpy(phase.astype(np.float32)).unsqueeze(0).to(self.device)
            if phase is not None
            else None
        )
        with torch.no_grad():
            normalized_action = self.model(
                observation_tensor,
                joint_tensor,
                goal_tensor,
                phase_tensor,
            )
            if normalized_action.ndim == 2:
                normalized_action = normalized_action.unsqueeze(1)
            action = denormalize_action(
                normalized_action,
                self.normalization,
            )
        return action.squeeze(0).cpu().numpy().astype(np.float64)
