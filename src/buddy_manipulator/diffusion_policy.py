"""Conditional diffusion policy for RGB-D action chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from buddy_manipulator.behavior_cloning import (
    MpsSafeAdaptiveAvgPool2d,
    NormalizationStats,
    choose_device,
    denormalize_action,
    normalize_observation,
)


def cosine_beta_schedule(
    diffusion_steps: int,
    *,
    offset: float = 0.008,
) -> torch.Tensor:
    """Return the improved-DDPM cosine noise schedule."""
    if diffusion_steps <= 1:
        raise ValueError("diffusion_steps must be greater than one")
    steps = torch.arange(diffusion_steps + 1, dtype=torch.float64)
    cumulative = torch.cos(
        ((steps / diffusion_steps + offset) / (1.0 + offset)) * math.pi / 2.0
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).to(dtype=torch.float32)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    if embedding_dim < 4 or embedding_dim % 2:
        raise ValueError("embedding_dim must be an even integer of at least four")
    half_dim = embedding_dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(
            half_dim,
            device=timesteps.device,
            dtype=torch.float32,
        )
        / (half_dim - 1)
    )
    angles = timesteps.to(dtype=torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResidualMlpBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class DiffusionPolicy(nn.Module):
    """Denoise a complete action chunk conditioned on one RGB-D observation."""

    def __init__(
        self,
        *,
        image_channels: int = 4,
        state_dim: int = 6,
        action_dim: int = 6,
        action_horizon: int = 8,
        diffusion_steps: int = 50,
        condition_dim: int = 128,
        hidden_dim: int = 256,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action dimensions must be positive")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.residual_blocks = residual_blocks
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            MpsSafeAdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(64 * 2 * 2 + state_dim, condition_dim),
            nn.LayerNorm(condition_dim),
            nn.SiLU(),
        )
        self.timestep_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim),
        )
        flattened_action_dim = action_horizon * action_dim
        self.action_projection = nn.Linear(flattened_action_dim, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *(ResidualMlpBlock(hidden_dim) for _ in range(residual_blocks))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flattened_action_dim),
        )

        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))

    def encode_condition(
        self,
        observation: torch.Tensor,
        joint_position: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_encoder(observation)
        return self.condition_encoder(
            torch.cat([image_features, joint_position], dim=1)
        )

    def denoise(
        self,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_action.ndim != 3:
            raise ValueError("noisy_action must have shape (batch, horizon, action_dim)")
        time_embedding = sinusoidal_timestep_embedding(
            timesteps,
            self.condition_dim,
        )
        hidden = self.action_projection(noisy_action.flatten(start_dim=1))
        hidden = (
            hidden
            + self.condition_projection(condition)
            + self.timestep_encoder(time_embedding)
        )
        return self.output(self.blocks(hidden)).reshape_as(noisy_action)

    def forward(
        self,
        observation: torch.Tensor,
        joint_position: torch.Tensor,
        noisy_action: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoise(
            noisy_action,
            timesteps,
            self.encode_condition(observation, joint_position),
        )

    def add_noise(
        self,
        action: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.alphas_cumprod[timesteps].reshape(-1, 1, 1)
        return alpha.sqrt() * action + (1.0 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        observation: torch.Tensor,
        joint_position: torch.Tensor,
        initial_noise: torch.Tensor,
        *,
        inference_steps: int = 10,
        clip_sample: float = 5.0,
    ) -> torch.Tensor:
        """Generate an action chunk with deterministic DDIM sampling."""
        if not 1 <= inference_steps <= self.diffusion_steps:
            raise ValueError("inference_steps must be within the diffusion schedule")
        if initial_noise.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("initial_noise has incompatible action shape")
        condition = self.encode_condition(observation, joint_position)
        sample = initial_noise
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            inference_steps,
            device=sample.device,
        ).round().to(dtype=torch.long)
        schedule = torch.unique_consecutive(schedule)
        for index, timestep in enumerate(schedule):
            timestep_value = int(timestep.item())
            timesteps = torch.full(
                (sample.shape[0],),
                timestep_value,
                device=sample.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoise(sample, timesteps, condition)
            alpha = self.alphas_cumprod[timestep_value]
            predicted_action = (
                sample - (1.0 - alpha).sqrt() * predicted_noise
            ) / alpha.sqrt()
            predicted_action = predicted_action.clamp(-clip_sample, clip_sample)
            if index + 1 == len(schedule):
                sample = predicted_action
                continue
            previous_timestep = int(schedule[index + 1].item())
            previous_alpha = self.alphas_cumprod[previous_timestep]
            sample = (
                previous_alpha.sqrt() * predicted_action
                + (1.0 - previous_alpha).sqrt() * predicted_noise
            )
        return sample


@dataclass
class DiffusionPolicyRunner:
    model: DiffusionPolicy
    normalization: NormalizationStats
    device: torch.device
    inference_steps: int
    inference_seed: int
    generator: torch.Generator

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
    ) -> "DiffusionPolicyRunner":
        device = choose_device(device_name)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("policy_type") != "diffusion":
            raise ValueError("checkpoint is not a diffusion policy")
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
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        inference_seed = int(checkpoint.get("inference_seed", 0))
        return cls(
            model=model,
            normalization=NormalizationStats.from_dict(
                checkpoint["normalization"]
            ),
            device=device,
            inference_steps=int(checkpoint.get("inference_steps", 10)),
            inference_seed=inference_seed,
            generator=torch.Generator().manual_seed(inference_seed),
        )

    @property
    def action_horizon(self) -> int:
        return self.model.action_horizon

    def reset(self, seed: int | None = None) -> None:
        self.generator.manual_seed(self.inference_seed if seed is None else seed)

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
        initial_noise = torch.randn(
            (1, self.action_horizon, self.model.action_dim),
            generator=self.generator,
            dtype=torch.float32,
        ).to(self.device)
        normalized_action = self.model.sample(
            observation_tensor,
            joint_tensor,
            initial_noise,
            inference_steps=self.inference_steps,
        )
        action = denormalize_action(normalized_action, self.normalization)
        return action.squeeze(0).cpu().numpy().astype(np.float64)
