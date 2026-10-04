"""The answers the core gives: `Where`, `Size` and `Ask`, and the `Answer` a robot sends back.

A `Where` is what any robot gets for "here", "that one" or "from here to here": a point, box
or path in the robot's own frame, with an uncertainty and a record of the evidence. A `Size`
is "this big" or "small" in mm. An `Ask` is one question, when the core will not guess.

These are wire types (JSON in and out). Each carries `version: "1.0"`. Version 1.x may only
add optional fields and new enum values; readers ignore fields they don't know (plan 4.10).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deixis.geometry import Cov2
from deixis.types import Hold, Word

#: The schema version of every wire type (separate from the package version).
WIRE_VERSION = "1.0"

Vec2 = tuple[float, float]
Box4 = tuple[float, float, float, float]

Source = Literal[
    "gesture",
    "gesture+vlm",
    "vlm",
    "verbal_preset",
    "object_ref",
    "geometry",
    "default",
    "adjust",
]
SizeSource = Literal["gesture", "verbal_preset", "object_ref", "default", "adjust"]
AskReason = Literal[
    "no_gesture",
    "ambiguous",
    "unsteady",
    "no_size",
    "no_object",
    "ambiguous_object",
    "unknown_label",
    "unknown_place",
    "no_room",
    "outside_area",
    "no_previous",
    "unsupported",
]

_ZERO_COV: Cov2 = ((0.0, 0.0), (0.0, 0.0))


class Wire(BaseModel):
    """Base of the wire types: a version, and unknown fields ignored (forward compatible)."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(default=WIRE_VERSION, description="Schema version, 1.x.")

    @field_validator("version")
    @classmethod
    def _major_one(cls, v: str) -> str:
        if v.split(".")[0] != "1":
            raise ValueError(f"version {v} is not 1.x")
        return v


class Where(Wire):
    """A place in the robot's frame: a point, a box or a path."""

    frame: str = Field(description="The profile's frame; every coordinate is in it.")
    kind: Literal["point", "box", "path"]
    mean_mm: Vec2 | None = Field(
        default=None,
        description="The point, or the centre of a box, or the mean of a path's points. "
        "Always set in output; filled in for boxes and paths.",
    )
    box_mm: Box4 | None = Field(
        default=None, description="x, y, w, h with (x, y) the minimum corner."
    )
    path_mm: list[Vec2] | None = Field(default=None, description="Two or more points.")
    z_mm: float | None = Field(
        default=None, description="Height above the frame's surface, when known ('on that')."
    )
    cov_mm2: Cov2 = Field(default=_ZERO_COV, description="Uncertainty of mean_mm, mm^2.")
    thing: str | None = Field(default=None, description="The thing's id, for an object.")
    label: str | None = Field(default=None, description="The thing's label, for an object.")
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description='How it was found, e.g. {"rule": "nearest", "word": "here", '
        '"hold_t": [11.9, 12.2]}.',
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
        if self.kind == "path":
            if self.path_mm is None or len(self.path_mm) < 2:
                raise ValueError("a path needs at least two points")
            if self.mean_mm is None:
                n = len(self.path_mm)
                self.mean_mm = (
                    sum(p[0] for p in self.path_mm) / n,
                    sum(p[1] for p in self.path_mm) / n,
                )
        return self


class Size(Wire):
    """A size: it sets the larger side of the thing's box."""

    mm: float = Field(gt=0, description="The larger side of the thing's box, mm.")
    side: Literal["larger"] = "larger"
    source: SizeSource
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class Ask(Wire):
    """One question for the person, when the core will not guess."""

    reason: AskReason
    question: str = Field(description="From the profile's asks, with the labels filled in.")
    call: int | None = Field(default=None, description="Which call it is about.")
    arg: str | None = Field(default=None, description="Which argument of that call.")
    candidates: list[Where] = Field(
        default_factory=list, description="The places or things to choose from."
    )
    options: list[str] = Field(
        default_factory=list, description="Short names for the candidates or choices."
    )
    word: Word | None = Field(default=None, description="The word it is about.")


class Answer(Wire):
    """A reply to an `Ask`, or an argument already resolved, sent back with the request.

    Exactly one of: `candidate` (an index into the earlier Ask's candidates, which the robot
    re-sends as `where` or `thing`), `where`, `size`, `thing` (a thing's id) or `hold` (the
    person pointed again; it replaces the pick for the pointing words in this argument).
    """

    call: int = Field(default=0, ge=0)
    arg: str = Field(min_length=1)
    candidate: int | None = Field(default=None, ge=0)
    where: Where | None = None
    size: Size | None = None
    thing: str | None = None
    hold: Hold | None = None

    @model_validator(mode="after")
    def _one_value(self) -> Answer:
        given = [k for k in ("where", "size", "thing", "hold") if getattr(self, k) is not None]
        if len(given) > 1:
            raise ValueError(f"an answer carries one value, not {given}")
        if not given and self.candidate is None:
            raise ValueError("an answer needs a candidate, where, size, thing or hold")
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
    "no_size.range": "That size looks wrong to me. How big should it be?",
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
