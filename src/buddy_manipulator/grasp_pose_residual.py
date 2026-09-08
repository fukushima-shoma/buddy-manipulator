"""Bounded learned pose corrections around the RGB-D grasp detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from buddy_manipulator.goal_task import ManipulationGoal, detect_goal_object
from buddy_manipulator.sim_camera import RgbdFrame
from buddy_manipulator.vision import WorldDetection


GRASP_POSE_FEATURE_DIM = 8
GRASP_POSE_RESIDUAL_DIM = 3


@dataclass(frozen=True)
class GraspPoseFeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "GraspPoseFeatureStats":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
        )


def grasp_pose_features(
    frame: RgbdFrame,
    goal: ManipulationGoal,
) -> tuple[np.ndarray, WorldDetection]:
    """Encode the analytical pose, image geometry, and selected object color."""
    detection = detect_goal_object(frame, goal)
    height, width = frame.rgb.shape[:2]
    pixel = detection.pixel
    features = np.asarray(
        [
            detection.x,
            detection.y,
            detection.z,
            pixel.center_u / max(1, width - 1) * 2.0 - 1.0,
            pixel.center_v / max(1, height - 1) * 2.0 - 1.0,
            pixel.area / float(height * width),
            float(goal.object_color == "red"),
            float(goal.object_color == "purple"),
        ],
        dtype=np.float32,
    )
    return features, detection


def compute_feature_stats(features: np.ndarray) -> GraspPoseFeatureStats:
    if features.ndim != 2 or features.shape[1] != GRASP_POSE_FEATURE_DIM:
        raise ValueError("features must have shape (samples, 8)")
    std = np.maximum(features.std(axis=0), np.float32(1e-6))
    return GraspPoseFeatureStats(
        mean=features.mean(axis=0).astype(np.float32),
        std=std.astype(np.float32),
    )


class GraspPoseResidualPolicy(nn.Module):
    """Predict XYZ corrections in millimeters from analytical pose features."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden dimension must be positive")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(GRASP_POSE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, GRASP_POSE_RESIDUAL_DIM),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != GRASP_POSE_FEATURE_DIM:
            raise ValueError("features must have shape (batch, 8)")
        return self.network(features)


@dataclass
class GraspPoseResidualRunner:
    model: GraspPoseResidualPolicy
    feature_stats: GraspPoseFeatureStats
    device: torch.device
    blend: float = 1.0
    maximum_correction_m: tuple[float, float, float] = (0.015, 0.015, 0.015)
    last_residual_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be between zero and one")
        if any(value <= 0 for value in self.maximum_correction_m):
            raise ValueError("maximum corrections must be positive")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
        blend: float | None = None,
    ) -> "GraspPoseResidualRunner":
        from buddy_manipulator.behavior_cloning import choose_device

        device = choose_device(device_name)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("policy_type") != "grasp_pose_residual":
            raise ValueError("checkpoint is not a grasp pose residual policy")
        config = checkpoint["model_config"]
        model = GraspPoseResidualPolicy(
            hidden_dim=int(config.get("hidden_dim", 64))
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            feature_stats=GraspPoseFeatureStats.from_dict(
                checkpoint["feature_stats"]
            ),
            device=device,
            blend=float(checkpoint["blend"] if blend is None else blend),
            maximum_correction_m=tuple(
                float(value) for value in checkpoint["maximum_correction_m"]
            ),
        )

    def predict_detection(
        self,
        frame: RgbdFrame,
        goal: ManipulationGoal,
    ) -> WorldDetection:
        features, baseline = grasp_pose_features(frame, goal)
        normalized = (features - self.feature_stats.mean) / self.feature_stats.std
        inputs = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            residual_mm = self.model(inputs).squeeze(0).cpu().numpy()
        limits = np.asarray(self.maximum_correction_m, dtype=np.float32)
        residual_m = np.clip(residual_mm / 1000.0, -limits, limits)
        residual_m *= self.blend
        self.last_residual_m = residual_m.astype(np.float64)
        corrected = np.asarray(baseline.position) + residual_m
        return WorldDetection(
            pixel=baseline.pixel,
            x=float(corrected[0]),
            y=float(corrected[1]),
            z=float(corrected[2]),
            depth=baseline.depth,
        )
