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


def capture_rgbd(
    model: Any,
    data: Any,
    *,
    camera_name: str = "workspace_camera",
    width: int = 640,
    height: int = 480,
) -> RgbdFrame:
    """Render aligned RGB and metric-depth images from a named camera."""
    mujoco = _mujoco()
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise KeyError(f"unknown camera: {camera_name}")

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_name)
        depth = renderer.render().copy()
    finally:
        renderer.close()

    return RgbdFrame(
        rgb=rgb,
        depth=depth,
        vertical_fov_degrees=float(model.cam_fovy[camera_id]),
        camera_position=np.asarray(data.cam_xpos[camera_id]).copy(),
        camera_rotation=np.asarray(data.cam_xmat[camera_id]).reshape(3, 3).copy(),
    )
