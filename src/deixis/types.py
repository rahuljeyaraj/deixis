"""Timed inputs to the resolver. All times are seconds on the shared monotonic clock."""

from __future__ import annotations

from dataclasses import dataclass

from deixis.geometry import Cov2, Point


@dataclass(frozen=True)
class Word:
    """One recognised word with its start and end time."""

    text: str
    t0: float
    t1: float
    confidence: float | None = None


@dataclass(frozen=True)
class Hold:
    """A period when a tracked point (fingertip, foot) stayed still in a pointing pose."""

    t0: float
    t1: float
    mean: Point  # mm, in the frame named by `frame`
    spread: Point  # standard deviation in x and y, mm
    frame: str = ""  # the frame named by the profile
    source: str = "pointer"  # e.g. "finger", "feet", "gaze", "click"
    cov_mm2: Cov2 | None = None  # if set, used instead of the spread plus the base sigma

    @property
    def mid(self) -> float:
        return (self.t0 + self.t1) / 2
