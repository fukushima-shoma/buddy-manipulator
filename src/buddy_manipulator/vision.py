"""Color detection and pinhole-camera projection for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PixelDetection:
    center_u: float
    center_v: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    area: int
    mask: np.ndarray

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class WorldDetection:
    pixel: PixelDetection
    x: float
    y: float
    z: float
    depth: float

    @property
    def position(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


def detect_colored_object(
    rgb: np.ndarray,
    *,
    color: str,
    min_value: int = 100,
    dominance_ratio: float = 1.45,
    min_area: int = 20,
) -> PixelDetection | None:
    """Detect a red or blue object using a deterministic color rule."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    channel_indices = {"red": 0, "blue": 2}
    if color not in (*channel_indices, "purple"):
        raise ValueError(f"unsupported object color: {color}")

    channels = rgb.astype(np.float32)
    if color == "purple":
        red, green, blue = (channels[:, :, index] for index in range(3))
        mask = (
            (red >= min_value)
            & (blue >= min_value)
            & (red >= dominance_ratio * green)
            & (blue >= dominance_ratio * green)
        )
    else:
        selected = channels[:, :, channel_indices[color]]
        other_indices = [
            index for index in range(3) if index != channel_indices[color]
        ]
        mask = (
            (selected >= min_value)
            & (selected >= dominance_ratio * channels[:, :, other_indices[0]])
            & (selected >= dominance_ratio * channels[:, :, other_indices[1]])
        )
    rows, columns = np.nonzero(mask)
    if columns.size < min_area:
        return None

    return PixelDetection(
        center_u=float(columns.mean()),
        center_v=float(rows.mean()),
        x_min=int(columns.min()),
        y_min=int(rows.min()),
        x_max=int(columns.max()),
        y_max=int(rows.max()),
        area=int(columns.size),
        mask=mask,
    )


def detect_red_object(
    rgb: np.ndarray,
    *,
    min_red: int = 100,
    dominance_ratio: float = 1.45,
    min_area: int = 20,
) -> PixelDetection | None:
    """Backward-compatible red-object detector."""
    return detect_colored_object(
        rgb,
        color="red",
        min_value=min_red,
        dominance_ratio=dominance_ratio,
        min_area=min_area,
    )


def project_pixels_to_world(
    columns: np.ndarray,
    rows: np.ndarray,
    depths: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    vertical_fov_degrees: float,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> np.ndarray:
    """Project image pixels and metric depth into MuJoCo world coordinates."""
    focal = 0.5 * image_height / math.tan(
        math.radians(vertical_fov_degrees) * 0.5
    )
    center_u = (image_width - 1) * 0.5
    center_v = (image_height - 1) * 0.5
    camera_points = np.stack(
        (
            (columns - center_u) * depths / focal,
            -(rows - center_v) * depths / focal,
            -depths,
        ),
        axis=-1,
    )
    return camera_points @ camera_rotation.T + camera_position


def locate_detection_in_world(
    detection: PixelDetection,
    depth_image: np.ndarray,
    *,
    vertical_fov_degrees: float,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
) -> WorldDetection:
    """Estimate a robust 3D object location from all valid detected pixels."""
    if depth_image.shape != detection.mask.shape:
        raise ValueError("depth image and detection mask must have the same shape")

    rows, columns = np.nonzero(detection.mask)
    depths = depth_image[rows, columns].astype(np.float64)
    valid = np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        raise ValueError("no valid depth values inside the detection")

    depths = depths[valid]
    rows = rows[valid].astype(np.float64)
    columns = columns[valid].astype(np.float64)
    points = project_pixels_to_world(
        columns,
        rows,
        depths,
        image_width=depth_image.shape[1],
        image_height=depth_image.shape[0],
        vertical_fov_degrees=vertical_fov_degrees,
        camera_position=np.asarray(camera_position, dtype=np.float64),
        camera_rotation=np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3),
    )
    position = np.median(points, axis=0)
    return WorldDetection(
        pixel=detection,
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        depth=float(np.median(depths)),
    )
