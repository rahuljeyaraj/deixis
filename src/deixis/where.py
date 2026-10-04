"""The robot-agnostic answer to "where did the person mean?".

A `Where` is what the resolver hands to any robot: a point, box or path in a named frame,
with an uncertainty and a record of the evidence. See plan section 4.3.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Vec2 = tuple[float, float]
Cov2 = tuple[tuple[float, float], tuple[float, float]]

Source = Literal["gesture", "gesture+vlm", "vlm", "verbal_preset", "object_ref", "geometry"]


class Where(BaseModel):
    frame: str = Field(description="Frame of all coordinates, e.g. 'paper', 'floor', 'robot_base'.")
    kind: Literal["point", "box", "path"]
    mean_mm: Vec2 | None = Field(
        default=None, description="Point location, or the centre of a box. Filled in for boxes."
    )
    box_mm: tuple[float, float, float, float] | None = Field(
        default=None, description="x, y, w, h with (x, y) the minimum corner."
    )
    path_mm: list[Vec2] | None = Field(default=None, description="At least two points.")
    cov_mm2: Cov2 = ((0.0, 0.0), (0.0, 0.0))
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description='E.g. {"word": "here", "word_t": 12.3, "hold_t": [11.9, 12.2]}.',
    )

    @model_validator(mode="after")
    def _check_kind(self) -> Where:
        if self.kind == "point" and self.mean_mm is None:
            raise ValueError("a point needs mean_mm")
        if self.kind == "box":
            if self.box_mm is None:
                raise ValueError("a box needs box_mm")
            x, y, w, h = self.box_mm
            if w < 0 or h < 0:
                raise ValueError("box width and height must be non-negative")
            if self.mean_mm is None:
                self.mean_mm = (x + w / 2, y + h / 2)
        if self.kind == "path" and (self.path_mm is None or len(self.path_mm) < 2):
            raise ValueError("a path needs at least two points")
        return self


#: Default question texts by `Ask` reason (English). A profile overrides any of them in
#: `asks`. A key "reason.variant" is a more specific text for one case of a reason; it falls
#: back to "reason". {placeholders} are filled from the case (label, name, options, a, b).
ASK_TEXTS: dict[str, str] = {
    "no_gesture": "Where do you mean? Please point.",
    "no_gesture.pointed": "Where should it go? Please point.",
    "ambiguous": "Did you mean this place or that one? Please point again.",
    "unsteady": "I couldn't tell where you pointed. Hold still a moment.",
    "no_size": "How big? Show me with your fingers or hands.",
    "no_object": "I don't see anything there. Which one do you mean?",
    "no_object.empty": "I don't see anything yet. What do you mean?",
    "ambiguous_object": "The {options}?",
    "unknown_label": "I don't see a {label}. Which one do you mean?",
    "unknown_place": "Where is '{name}'? Please point at it.",
    "no_room": "There is no free space left.",
    "outside_area": "I can't reach there. Please point somewhere else.",
    "no_previous": "There is nothing to change right now.",
    "unsupported": "I can't do that yet.",
}


class _Blank(dict):  # type: ignore[type-arg]
    def __missing__(self, key: str) -> str:
        return ""


def ask_text(
    reason: str, variant: str | None = None, asks: Mapping[str, str] | None = None, **fill: Any
) -> str:
    """The question for a reason: the profile's text first, then the default."""
    asks = asks or {}
    keys = ([f"{reason}.{variant}"] if variant else []) + [reason]
    text = next((asks[k] for k in keys if k in asks), None)
    if text is None:
        text = next((ASK_TEXTS[k] for k in keys if k in ASK_TEXTS), "Sorry, what do you mean?")
    return text.format_map(_Blank({k: str(v) for k, v in fill.items()}))


def where_json_schema() -> dict[str, Any]:
    """JSON Schema for `Where`, for other languages and robots."""
    return Where.model_json_schema()
