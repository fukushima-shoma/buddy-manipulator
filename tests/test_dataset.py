import json

import numpy as np

from buddy_manipulator.collect_demos import next_episode_index
from buddy_manipulator.dataset import save_episode, validate_episode


def example_arrays(samples: int = 3) -> dict[str, np.ndarray]:
    return {
        "timestamp": np.arange(samples, dtype=np.float64) * 0.2,
        "rgb": np.zeros((samples, 12, 16, 3), dtype=np.uint8),
        "depth": np.ones((samples, 12, 16), dtype=np.float32),
        "joint_position": np.zeros((samples, 6), dtype=np.float32),
        "action": np.zeros((samples, 6), dtype=np.float32),
        "object_position": np.zeros((samples, 3), dtype=np.float32),
    }


def test_saved_episode_passes_validation(tmp_path) -> None:
    data_path, metadata_path = save_episode(
        tmp_path,
        4,
        example_arrays(),
        success=True,
        block_start_position=(0.3, 0.08, 0.025),
    )

    assert validate_episode(data_path, metadata_path) == []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["sample_count"] == 3
    assert metadata["success"] is True
    assert metadata["source"] == "scripted"
    assert metadata["termination"] == "completed"


def test_validator_detects_sample_count_mismatch(tmp_path) -> None:
    data_path, metadata_path = save_episode(
        tmp_path,
        0,
        example_arrays(),
        success=False,
        block_start_position=(0.3, 0.08, 0.025),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["sample_count"] = 99
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert "metadata sample_count mismatch" in validate_episode(
        data_path, metadata_path
    )


def test_next_episode_index_does_not_overwrite_existing_data(tmp_path) -> None:
    (tmp_path / "episode_00002.npz").touch()
    (tmp_path / "episode_00007.npz").touch()
    assert next_episode_index(tmp_path) == 8
