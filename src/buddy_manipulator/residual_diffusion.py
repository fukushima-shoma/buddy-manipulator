"""Residual diffusion policy anchored by a frozen behavior-cloning prior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from buddy_manipulator.behavior_cloning import (
    BehaviorCloningPolicy,
    NormalizationStats,
    choose_device,
    denormalize_action,
    normalize_observation,
)
from buddy_manipulator.diffusion_policy import DiffusionPolicy


def load_bc_prior(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BehaviorCloningPolicy, NormalizationStats, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported BC checkpoint format")
    config = checkpoint["model_config"]
    model = BehaviorCloningPolicy(
        image_channels=int(config["image_channels"]),
        state_dim=int(config["state_dim"]),
        action_horizon=int(config.get("action_horizon", 1)),
        use_object_features=bool(config.get("use_object_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    return model, NormalizationStats.from_dict(checkpoint["normalization"]), config


def _vector(
    values: np.ndarray,
    reference: torch.Tensor,
) -> torch.Tensor:
    return torch.as_tensor(
        values,
        dtype=reference.dtype,
        device=reference.device,
    )


@torch.no_grad()
def predict_prior_in_normalization(
    base_model: BehaviorCloningPolicy,
    base_normalization: NormalizationStats,
    target_normalization: NormalizationStats,
    observation: torch.Tensor,
    joint_position: torch.Tensor,
) -> torch.Tensor:
    """Run the BC prior when the batch uses another set of normalization stats."""
    base_observation = observation.clone()
    raw_depth = (
        observation[:, 3]
        * float(target_normalization.depth_std)
        + float(target_normalization.depth_mean)
    )
    base_observation[:, 3] = (
        raw_depth - float(base_normalization.depth_mean)
    ) / float(base_normalization.depth_std)
    raw_joint = (
        joint_position * _vector(target_normalization.joint_std, joint_position)
        + _vector(target_normalization.joint_mean, joint_position)
    )
    base_joint = (
        raw_joint - _vector(base_normalization.joint_mean, joint_position)
    ) / _vector(base_normalization.joint_std, joint_position)
    base_action = base_model(base_observation, base_joint)
    if base_action.ndim == 2:
        base_action = base_action.unsqueeze(1)
    physical_action = denormalize_action(base_action, base_normalization)
    return (
        physical_action - _vector(target_normalization.action_mean, physical_action)
    ) / _vector(target_normalization.action_std, physical_action)


@dataclass
class ResidualDiffusionRunner:
    model: DiffusionPolicy
    base_model: BehaviorCloningPolicy
    normalization: NormalizationStats
    base_normalization: NormalizationStats
    residual_mean: torch.Tensor
    residual_std: torch.Tensor
    device: torch.device
    inference_steps: int
    inference_seed: int
    initial_noise_scale: float
    residual_blend: float
    generator: torch.Generator

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
    ) -> "ResidualDiffusionRunner":
        device = choose_device(device_name)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("policy_type") != "residual_diffusion":
            raise ValueError("checkpoint is not a residual diffusion policy")
        config = checkpoint["model_config"]
        model = DiffusionPolicy(
            image_channels=int(config["image_channels"]),
            state_dim=int(config["state_dim"]),
            action_dim=int(config["action_dim"]),
            action_horizon=int(config["action_horizon"]),
            diffusion_steps=int(config["diffusion_steps"]),
            condition_dim=int(config["condition_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            residual_blocks=int(config["residual_blocks"]),
            prior_action_dim=int(config["prior_action_dim"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        base_config = checkpoint["base_model_config"]
        base_model = BehaviorCloningPolicy(
            image_channels=int(base_config["image_channels"]),
            state_dim=int(base_config["state_dim"]),
            action_horizon=int(base_config.get("action_horizon", 1)),
            use_object_features=bool(base_config.get("use_object_features", False)),
        ).to(device)
        base_model.load_state_dict(checkpoint["base_model_state_dict"])
        base_model.eval().requires_grad_(False)
        inference_seed = int(checkpoint.get("inference_seed", 0))
        return cls(
            model=model,
            base_model=base_model,
            normalization=NormalizationStats.from_dict(checkpoint["normalization"]),
            base_normalization=NormalizationStats.from_dict(
                checkpoint["base_normalization"]
            ),
            residual_mean=torch.as_tensor(
                checkpoint["residual_mean"],
                dtype=torch.float32,
                device=device,
            ),
            residual_std=torch.as_tensor(
                checkpoint["residual_std"],
                dtype=torch.float32,
                device=device,
            ),
            device=device,
            inference_steps=int(checkpoint.get("inference_steps", 25)),
            inference_seed=inference_seed,
            initial_noise_scale=float(checkpoint.get("initial_noise_scale", 0.0)),
            residual_blend=float(checkpoint.get("residual_blend", 1.0)),
            generator=torch.Generator().manual_seed(inference_seed),
        )

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    def reset(self, seed: int | None = None) -> None:
        self.generator.manual_seed(self.inference_seed if seed is None else seed)

    def configure_sampling(
        self,
        *,
        inference_steps: int | None = None,
        initial_noise_scale: float | None = None,
        residual_blend: float | None = None,
    ) -> None:
        if inference_steps is not None:
            if not 1 <= inference_steps <= self.model.diffusion_steps:
                raise ValueError("inference_steps must be within the diffusion schedule")
            self.inference_steps = inference_steps
        if initial_noise_scale is not None:
            if initial_noise_scale < 0.0:
                raise ValueError("initial_noise_scale must be non-negative")
            self.initial_noise_scale = initial_noise_scale
        if residual_blend is not None:
            if not 0.0 <= residual_blend <= 1.0:
                raise ValueError("residual_blend must be between zero and one")
            self.residual_blend = residual_blend

    def predict(self, rgb: np.ndarray, depth: np.ndarray, joint_position: np.ndarray) -> np.ndarray:
        return self.predict_chunk(rgb, depth, joint_position)[0]

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
        observation_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        joint_tensor = torch.from_numpy(normalized_joint).unsqueeze(0).to(self.device)
        prior_action = predict_prior_in_normalization(
            self.base_model,
            self.base_normalization,
            self.normalization,
            observation_tensor,
            joint_tensor,
        )
        initial_noise = torch.randn(
            (1, self.action_horizon, self.model.action_dim),
            generator=self.generator,
            dtype=torch.float32,
        ).mul_(self.initial_noise_scale).to(self.device)
        residual = self.model.sample(
            observation_tensor,
            joint_tensor,
            initial_noise,
            inference_steps=self.inference_steps,
            prior_action=prior_action,
        )
        residual = residual * self.residual_std + self.residual_mean
        normalized_action = prior_action + self.residual_blend * residual
        action = denormalize_action(normalized_action, self.normalization)
        return action.squeeze(0).cpu().numpy().astype(np.float64)
