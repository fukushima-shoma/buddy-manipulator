"""RGB-D behavior cloning data pipeline and policy network."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class EpisodePath:
    data_path: Path
    metadata_path: Path

    @property
    def name(self) -> str:
        return self.data_path.stem


@dataclass(frozen=True)
class EpisodeData:
    name: str
    rgb: np.ndarray
    depth: np.ndarray
    joint_position: np.ndarray
    action: np.ndarray

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
        episodes.append(EpisodePath(data_path, metadata_path))
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


def load_episodes(episode_paths: Sequence[EpisodePath]) -> list[EpisodeData]:
    episodes = []
    for episode_path in episode_paths:
        with np.load(episode_path.data_path) as arrays:
            episodes.append(
                EpisodeData(
                    name=episode_path.name,
                    rgb=arrays["rgb"].copy(),
                    depth=arrays["depth"].astype(np.float32, copy=True),
                    joint_position=arrays["joint_position"].astype(
                        np.float32, copy=True
                    ),
                    action=arrays["action"].astype(np.float32, copy=True),
                )
            )
    return episodes


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
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.episodes = list(episodes)
        self.normalization = normalization
        self.action_horizon = action_horizon
        self._sample_index = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(self.episodes)
            for frame_index in range(episode.sample_count)
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
        return {
            "observation": torch.from_numpy(observation),
            "joint_position": torch.from_numpy(joint.astype(np.float32)),
            "action": torch.from_numpy(action.astype(np.float32)),
        }


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


class BehaviorCloningPolicy(nn.Module):
    """Small CNN policy for RGB-D and proprioceptive observations."""

    def __init__(
        self,
        *,
        image_channels: int = 4,
        state_dim: int = 6,
        action_horizon: int = 1,
    ) -> None:
        super().__init__()
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self.action_horizon = action_horizon
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
        self.action_head = nn.Sequential(
            nn.Linear(64 * 2 * 2 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6 * action_horizon),
        )

    def forward(
        self, observation: torch.Tensor, joint_position: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.image_encoder(observation)
        action = self.action_head(
            torch.cat([image_features, joint_position], dim=1)
        )
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
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            normalization=NormalizationStats.from_dict(
                checkpoint["normalization"]
            ),
            device=device,
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

    def predict_chunk(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        joint_position: np.ndarray,
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
        joint_tensor = torch.from_numpy(normalized_joint).unsqueeze(0).to(
            self.device
        )
        with torch.no_grad():
            normalized_action = self.model(observation_tensor, joint_tensor)
            if normalized_action.ndim == 2:
                normalized_action = normalized_action.unsqueeze(1)
            action = denormalize_action(
                normalized_action,
                self.normalization,
            )
        return action.squeeze(0).cpu().numpy().astype(np.float64)
