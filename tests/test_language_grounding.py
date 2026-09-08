import pytest

from buddy_manipulator.language_grounding import (
    GroundingError,
    LexiconGoalGrounder,
    normalize_instruction,
)


@pytest.mark.parametrize(
    ("instruction", "object_color", "target_color"),
    [
        ("Place the red block in the green zone.", "red", "green"),
        ("Move the violet cube onto the yellow target", "purple", "yellow"),
        ("赤いブロックを緑色のエリアに置いて", "red", "green"),
        ("紫のキューブを黄色のターゲットへ運んで", "purple", "yellow"),
    ],
)
def test_bilingual_grounder_maps_supported_goals(
    instruction: str, object_color: str, target_color: str
) -> None:
    result = LexiconGoalGrounder().ground(instruction)

    assert result.goal.object_color == object_color
    assert result.goal.target_color == target_color
    assert result.confidence == 1.0
    assert result.method == "bilingual_lexicon_v1"


@pytest.mark.parametrize(
    "instruction",
    [
        "Move the block to the green zone",
        "Move the red and purple blocks to yellow",
        "赤いブロックを箱に入れて",
        "",
    ],
)
def test_bilingual_grounder_rejects_missing_or_ambiguous_goals(
    instruction: str,
) -> None:
    with pytest.raises(GroundingError):
        LexiconGoalGrounder().ground(instruction)


def test_normalization_handles_full_width_text_and_whitespace() -> None:
    assert normalize_instruction("  ＲＥＤ   block  ") == "red block"
