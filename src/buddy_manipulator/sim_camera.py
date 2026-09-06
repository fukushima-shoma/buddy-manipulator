"""MuJoCo RGB-D camera adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from buddy_manipulator.simulation import _mujoco


@dataclass(frozen=True)
class RgbdFrame:
    rgb: np.ndarray
    depth: np.ndarray
    vertical_fov_degrees: float
    camera_position: np.ndarray
    camera_rotation: np.ndarray


class RgbdCamera:
    """Reusable renderer for aligned RGB and metric-depth observations."""

    def __init__(
        self,
        model: Any,
        *,
        camera_name: str = "workspace_camera",
        width: int = 640,
        height: int = 480,
    ) -> None:
        mujoco = _mujoco()
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id < 0:
            raise KeyError(f"unknown camera: {camera_name}")
        self.model = model
        self.camera_name = camera_name
        self.camera_id = camera_id
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def capture(self, data: Any) -> RgbdFrame:
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=self.camera_name)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

        return RgbdFrame(
            rgb=rgb,
            depth=depth,
            vertical_fov_degrees=float(self.model.cam_fovy[self.camera_id]),
            camera_position=np.asarray(data.cam_xpos[self.camera_id]).copy(),
            camera_rotation=np.asarray(data.cam_xmat[self.camera_id])
            .reshape(3, 3)
            .copy(),
        )

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "RgbdCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def capture_rgbd(
    model: Any,
    data: Any,
    *,
    camera_name: str = "workspace_camera",
    width: int = 640,
    height: int = 480,
) -> RgbdFrame:
    """Render one aligned RGB-D observation."""
    with RgbdCamera(
        model,
        camera_name=camera_name,
        width=width,
        height=height,
    ) as camera:
        return camera.capture(data)
