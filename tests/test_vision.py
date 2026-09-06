import numpy as np
import pytest

from buddy_manipulator.vision import (
    PixelDetection,
    detect_red_object,
    locate_detection_in_world,
    project_pixels_to_world,
)


def test_detect_red_object_returns_bbox_and_center() -> None:
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    rgb[20:40, 50:80] = (220, 20, 10)

    detection = detect_red_object(rgb)

    assert detection is not None
    assert detection.bbox == (50, 20, 79, 39)
    assert detection.center_u == pytest.approx(64.5)
    assert detection.center_v == pytest.approx(29.5)
    assert detection.area == 600


def test_detect_red_object_ignores_small_noise() -> None:
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    rgb[3, 4] = (255, 0, 0)
    assert detect_red_object(rgb) is None


def test_project_center_pixel_along_camera_view_axis() -> None:
    points = project_pixels_to_world(
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.5]),
        image_width=3,
        image_height=3,
        vertical_fov_degrees=60.0,
        camera_position=np.array([0.0, 0.0, 1.0]),
        camera_rotation=np.eye(3),
    )
    assert points[0] == pytest.approx((0.0, 0.0, 0.5))


def test_locate_detection_uses_depth_and_camera_pose() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    detection = PixelDetection(1.0, 1.0, 1, 1, 1, 1, 1, mask)
    depth = np.ones((3, 3), dtype=float) * 0.5

    world = locate_detection_in_world(
        detection,
        depth,
        vertical_fov_degrees=60.0,
        camera_position=np.array([0.2, -0.1, 1.0]),
        camera_rotation=np.eye(3),
    )

    assert world.position == pytest.approx((0.2, -0.1, 0.5))
    assert world.depth == pytest.approx(0.5)
