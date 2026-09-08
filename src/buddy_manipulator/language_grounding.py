"""Natural-language grounding into the Phase 5 structured goal space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Protocol
import unicodedata

from buddy_manipulator.goal_task import ManipulationGoal


class GroundingError(ValueError):
    """Raised when an instruction cannot be mapped to one safe unique goal."""


@dataclass(frozen=True)
class GroundingResult:
    goal: ManipulationGoal
    original_instruction: str
    normalized_instruction: str
    object_evidence: tuple[str, ...]
    target_evidence: tuple[str, ...]
    method: str = "bilingual_lexicon_v1"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["goal"] = self.goal.to_dict()
        return values


class GoalGrounder(Protocol):
    def ground(self, instruction: str) -> GroundingResult:
        ...


OBJECT_ALIASES = {
    "red": ("red", "crimson", "赤い", "赤色", "赤"),
    "purple": ("purple", "violet", "紫色", "紫"),
}
TARGET_ALIASES = {
    "green": ("green", "緑色", "緑"),
    "yellow": ("yellow", "黄色", "黄"),
}


def normalize_instruction(instruction: str) -> str:
    normalized = unicodedata.normalize("NFKC", instruction).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _contains_alias(text: str, alias: str) -> bool:
    if alias.isascii():
        return re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text) is not None
    return alias in text


def _match_colors(
    text: str,
    aliases: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    matches = {}
    for color, candidates in aliases.items():
        evidence = tuple(alias for alias in candidates if _contains_alias(text, alias))
        if evidence:
            matches[color] = evidence
    return matches


@dataclass(frozen=True)
class LexiconGoalGrounder:
    """Conservative bilingual baseline that rejects uncertain commands."""

    def ground(self, instruction: str) -> GroundingResult:
        if not instruction.strip():
            raise GroundingError("instruction must not be empty")
        normalized = normalize_instruction(instruction)
        object_matches = _match_colors(normalized, OBJECT_ALIASES)
        target_matches = _match_colors(normalized, TARGET_ALIASES)
        if len(object_matches) != 1:
            found = ", ".join(object_matches) or "none"
            raise GroundingError(
                f"instruction must identify exactly one supported object color; found {found}"
            )
        if len(target_matches) != 1:
            found = ", ".join(target_matches) or "none"
            raise GroundingError(
                f"instruction must identify exactly one supported target color; found {found}"
            )
        object_color = next(iter(object_matches))
        target_color = next(iter(target_matches))
        return GroundingResult(
            goal=ManipulationGoal(object_color, target_color),
            original_instruction=instruction,
            normalized_instruction=normalized,
            object_evidence=object_matches[object_color],
            target_evidence=target_matches[target_color],
        )
