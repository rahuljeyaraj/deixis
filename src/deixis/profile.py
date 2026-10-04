"""The robot profile: everything robot-specific the core needs, given once.

A profile names the robot's frame and work area, says how its pointer behaves (how still is
"held", how noisy it is), how relations read ("left", "on"), the sizes, the question texts
and the robot's own actions. The core's code has no robot words in it; they all come from
here. Profiles are JSON files (`load_profile`); the three demos' profiles are copied in
`examples/profiles/`.

Units: millimetres and seconds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deixis.actions import STANDARD_ACTIONS, Action
from deixis.filters import OneEuroParams
from deixis.geometry import Box
from deixis.holds import HoldParams
from deixis.resolver import ResolverParams
from deixis.spans import SpanParams
from deixis.where import ASK_TEXTS, ask_text

PROFILE_VERSION = "1.0"

#: How relations read (plan 4.3, F8):
#: - "page": a picture seen from above; "above" is up the page (away from the viewer) and
#:   "on" means on the target's top edge.
#: - "surface": things standing on a surface; "above" and "on" mean stacked on top (with the
#:   height of the thing underneath), "below" is not supported, "behind" is away.
#: - "ground": like "surface", but nothing is stacked by "above": above/below are asked about.
ViewMode = Literal["page", "surface", "ground"]

Vec2 = tuple[float, float]
Box4 = tuple[float, float, float, float]

#: Words that can carry a pointing gesture ("so" as in "so big"). English only (L7).
DEMONSTRATIVES = ("here", "there", "this", "that", "these", "those", "so")

_AXES = {(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Viewer(_Model):
    """The person's right and away-from-them, as unit vectors along the frame's axes."""

    right: Vec2 = (1.0, 0.0)
    away: Vec2 = (0.0, 1.0)

    @model_validator(mode="after")
    def _axis_aligned(self) -> Viewer:
        r = (float(self.right[0]), float(self.right[1]))
        a = (float(self.away[0]), float(self.away[1]))
        if r not in _AXES or a not in _AXES:
            raise ValueError("viewer.right and viewer.away must be unit vectors along an axis")
        if r[0] * a[0] + r[1] * a[1] != 0:
            raise ValueError("viewer.right and viewer.away must be perpendicular")
        return self


class FilterSettings(_Model):
    """One-Euro smoothing of the pointer (Casiez et al., CHI 2012)."""

    min_cutoff: float = Field(default=1.0, gt=0)
    beta: float = Field(default=0.02, ge=0)
    d_cutoff: float = Field(default=1.0, gt=0)


class HoldSettings(_Model):
    """When a pointer counts as held still (see `deixis.holds.HoldParams`)."""

    speed_enter_mm_s: float = Field(default=20.0, gt=0)
    speed_exit_mm_s: float = Field(default=30.0, gt=0)
    min_duration_s: float = Field(default=0.15, ge=0)
    speed_window_s: float = Field(default=0.10, gt=0)
    max_gap_s: float = Field(default=0.12, ge=0)
    require_pointing: bool = True
    history_s: float = Field(default=120.0, gt=0)


class PointerSettings(_Model):
    source: str = Field(default="pointer", description="Label for holds: finger, feet, gaze, ...")
    filter: FilterSettings | None = Field(default_factory=FilterSettings)
    hold: HoldSettings = Field(default_factory=HoldSettings)
    base_sigma_mm: float = Field(default=1.5, ge=0, description="Added to a hold's spread.")
    max_spread_mm: float = Field(default=6.0, gt=0, description="More wander: ask 'unsteady'.")
    same_place_mm: float = Field(default=10.0, ge=0, description="Closer holds: one place.")


class SpanSettings(_Model):
    """Steady size gestures ("this big"); see `deixis.spans.SpanParams`."""

    max_rate_mm_s: float = Field(default=25.0, gt=0)
    window_s: float = Field(default=0.10, gt=0)
    min_duration_s: float = Field(default=0.20, ge=0)
    min_span_mm: float = Field(default=8.0, ge=0)
    max_gap_s: float = Field(default=0.12, ge=0)
    history_s: float = Field(default=30.0, gt=0)


class TimingSettings(_Model):
    """How words are paired with holds (plan 4.3, LR 3.4)."""

    before_s: float = Field(default=1.5, ge=0)
    after_s: float = Field(default=0.5, ge=0)
    no_word_before_s: float = Field(default=2.0, ge=0)
    time_sigma_s: float = Field(default=0.6, gt=0)
    ambiguity_ratio: float = Field(default=2.0, ge=1)
    size_overlap: float = Field(default=0.5, ge=0, le=1)


class SizeSettings(_Model):
    default_mm: float = Field(
        default=40.0, gt=0, description="When no size was said; also a point's box (F12)."
    )
    presets: dict[str, float | None] = Field(
        default_factory=dict, description="Size words in mm; null = the whole area."
    )
    synonyms: dict[str, str] = Field(
        default_factory=dict, description="Extra words for the presets, e.g. 'huge' -> 'big'."
    )
    min_mm: float = Field(default=0.0, ge=0)
    max_mm: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _synonyms_name_presets(self) -> SizeSettings:
        bad = sorted(v for v in self.synonyms.values() if v not in self.presets)
        if bad:
            raise ValueError(f"size synonyms point at unknown presets: {', '.join(bad)}")
        return self


class SceneSettings(_Model):
    near_mm: float = Field(default=20.0, ge=0, description="'that one': things this close.")
    ambiguous_mm: float = Field(default=6.0, ge=0, description="Two this close: ask.")
    gap_mm: float = Field(default=10.0, ge=0, description="The gap for 'next to'.")
    inside_fill: float = Field(default=0.7, gt=0, le=1)
    empty_margin_mm: float = Field(default=3.0, ge=0)
    grid_mm: float = Field(default=1.0, gt=0, description="Grid step for empty space.")
    label_ties: Literal["most_recent", "ask"] = "most_recent"


class AdjustSettings(_Model):
    step_mm: float = Field(default=5.0, gt=0, description="One 'a bit' step.")


class WordSettings(_Model):
    demonstratives: list[str] = Field(default_factory=lambda: list(DEMONSTRATIVES))

    @field_validator("demonstratives")
    @classmethod
    def _english_only(cls, v: list[str]) -> list[str]:
        v = [w.strip().lower() for w in v]
        extra = sorted(set(v) - set(DEMONSTRATIVES))
        if extra:  # the reference types list these words (L7: English only for now)
            raise ValueError(f"unknown demonstratives {extra}; v1 knows {list(DEMONSTRATIVES)}")
        return v


class Profile(_Model):
    """A robot's profile (plan 4.3)."""

    version: str = PROFILE_VERSION
    name: str = Field(min_length=1)
    frame: str = Field(min_length=1, description="The frame of every position.")
    area_mm: Box4 = Field(description="Where the robot can act: x, y, w, h.")
    view: ViewMode = "page"
    viewer: Viewer = Field(default_factory=Viewer)
    pointer: PointerSettings = Field(default_factory=PointerSettings)
    spans: SpanSettings | None = Field(default=None, description='null = no "this big".')
    timing: TimingSettings = Field(default_factory=TimingSettings)
    sizes: SizeSettings = Field(default_factory=SizeSettings)
    scene: SceneSettings = Field(default_factory=SceneSettings)
    adjust: AdjustSettings = Field(default_factory=AdjustSettings)
    outside: Literal["keep", "ask"] = Field(
        default="keep", description="A place outside the area: keep it (flagged) or ask."
    )
    asks: dict[str, str] = Field(
        default_factory=dict, description="Question texts by reason; defaults fill the rest."
    )
    words: WordSettings = Field(default_factory=WordSettings)
    actions: list[Action] = Field(default_factory=list)

    @field_validator("version")
    @classmethod
    def _major_one(cls, v: str) -> str:
        if v.split(".")[0] != "1":
            raise ValueError(f"profile version {v} is not 1.x")
        return v

    @field_validator("area_mm")
    @classmethod
    def _area(cls, v: Box4) -> Box4:
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("area_mm needs a positive width and height")
        return v

    @field_validator("asks")
    @classmethod
    def _known_reasons(cls, v: dict[str, str]) -> dict[str, str]:
        known = {k.split(".")[0] for k in ASK_TEXTS}
        bad = sorted(k for k in v if k.split(".")[0] not in known)
        if bad:
            raise ValueError(f"unknown ask reasons: {', '.join(bad)}")
        return v

    @model_validator(mode="after")
    def _unique_actions(self) -> Profile:
        names = [a.name for a in self.actions]
        std = {a.name for a in STANDARD_ACTIONS}
        if len(names) != len(set(names)):
            raise ValueError("action names must be unique")
        clash = sorted(set(names) & std)
        if clash:
            raise ValueError(f"actions {clash} are standard actions; don't redeclare them")
        return self

    # ------------------------------------------------------------------ views

    @property
    def area(self) -> Box:
        return Box(*self.area_mm)

    def all_actions(self) -> list[Action]:
        """The robot's actions plus the standard ones (adjust, confirm, cancel, ...)."""
        return [*self.actions, *STANDARD_ACTIONS]

    def action(self, name: str) -> Action | None:
        return next((a for a in self.all_actions() if a.name == name), None)

    def ask_text(self, reason: str, variant: str | None = None, **fill: Any) -> str:
        """The question for an `Ask` reason, with {placeholders} filled in."""
        return ask_text(reason, variant, self.asks, **fill)

    # ------------------------------------------------------------------ parameters

    def resolver_params(self) -> ResolverParams:
        t, p = self.timing, self.pointer
        return ResolverParams(
            before_s=t.before_s,
            after_s=t.after_s,
            no_word_before_s=t.no_word_before_s,
            time_sigma_s=t.time_sigma_s,
            ambiguity_ratio=t.ambiguity_ratio,
            same_place_mm=p.same_place_mm,
            base_sigma_mm=p.base_sigma_mm,
            max_spread_mm=p.max_spread_mm,
            size_overlap=t.size_overlap,
            asks=tuple(sorted(self.asks.items())),
        )

    def hold_params(self) -> HoldParams:
        return HoldParams(
            **self.pointer.hold.model_dump(), frame=self.frame, source=self.pointer.source
        )

    def span_params(self) -> SpanParams | None:
        return None if self.spans is None else SpanParams(**self.spans.model_dump())

    def filter_params(self) -> OneEuroParams | None:
        f = self.pointer.filter
        return None if f is None else OneEuroParams(**f.model_dump())


def load_profile(source: str | Path | Mapping[str, Any] | Profile) -> Profile:
    """A profile from a JSON file path, a dict, or a `Profile` (returned as is)."""
    if isinstance(source, Profile):
        return source
    if isinstance(source, Mapping):
        return Profile.model_validate(dict(source))
    text = Path(source).read_text(encoding="utf-8")
    return Profile.model_validate(json.loads(text))
