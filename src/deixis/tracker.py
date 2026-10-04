"""From pointer samples to holds and size gestures (plan 4.7, F13, F14).

`Tracker` is the streaming version, for a live display: feed one `Sample` per camera frame.
`track` runs a fresh tracker over a list of samples (what `resolve` does with
`Request.samples`): the same samples always give the same holds.

Per sample: the pointer is smoothed with the profile's One-Euro filter (reset when it is not
tracked), then fed to the hold detector; a size gesture (two points) is fed to the span
detector, if the profile allows size gestures. A sample's own covariance (gaze, a far ray) is
kept: a hold made from such samples carries their mean covariance plus its own spread.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import replace

from deixis.filters import OneEuroFilter2D
from deixis.geometry import Cov2, Point
from deixis.holds import HoldDetector
from deixis.profile import Profile
from deixis.resolver import SpanHold
from deixis.spans import SpanHoldDetector
from deixis.types import Hold, Sample


class Tracker:
    """Streaming holds and spans for one pointer. Not thread-safe; call from one thread."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        fp = profile.filter_params()
        self._filter = OneEuroFilter2D(fp) if fp is not None else None
        self.hold_detector = HoldDetector(profile.hold_params())
        sp = profile.span_params()
        self.span_detector = SpanHoldDetector(sp) if sp is not None else None
        self._covs: deque[tuple[float, Cov2]] = deque()
        self.point: Point | None = None  # the last smoothed point
        self.ended_hold: Hold | None = None  # a hold that ended with the last sample
        self.ended_span: SpanHold | None = None

    @property
    def speed(self) -> float | None:
        return self.hold_detector.speed

    def update(self, s: Sample) -> tuple[Hold | None, SpanHold | None]:
        """Add one sample. Returns the hold and the span that ended with it, if any."""
        if s.point is None:
            if self._filter is not None:
                self._filter.reset()
            self.point = None
        else:
            p = (float(s.point[0]), float(s.point[1]))
            self.point = self._filter(s.t, p) if self._filter is not None else p
            if s.cov_mm2 is not None:
                self._covs.append((s.t, s.cov_mm2))
                keep = self.hold_detector.params.history_s + 10.0
                while self._covs and self._covs[0][0] < s.t - keep:
                    self._covs.popleft()
        hold = self.hold_detector.update(s.t, self.point, s.pointing)
        span = None
        if self.span_detector is not None:
            if s.span is not None and s.span_kind is not None:
                (ax, ay), (bx, by) = s.span
                size = math.hypot(bx - ax, by - ay)
                center = ((ax + bx) / 2, (ay + by) / 2)
                span = self.span_detector.update(s.t, size, s.span_kind, center)
            else:
                span = self.span_detector.update(s.t, None)
        self.ended_hold = self._with_cov(hold) if hold is not None else None
        self.ended_span = span
        return self.ended_hold, span

    def current(self) -> Hold | None:
        """The hold in progress, if it is already long enough to count."""
        h = self.hold_detector.current()
        return self._with_cov(h) if h is not None else None

    def holds(
        self, t_from: float = -math.inf, t_to: float = math.inf, include_current: bool = True
    ) -> list[Hold]:
        """Finished holds (and the current one) overlapping [t_from, t_to]."""
        return [self._with_cov(h) for h in self.hold_detector.holds(t_from, t_to, include_current)]

    def spans(self, t_from: float = -math.inf, t_to: float = math.inf) -> list[SpanHold]:
        if self.span_detector is None:
            return []
        return self.span_detector.holds(t_from, t_to)

    def _with_cov(self, h: Hold) -> Hold:
        """A hold made of samples that knew their own uncertainty: their mean covariance
        plus the hold's spread."""
        if not self._covs:
            return h
        inside = [c for t, c in self._covs if h.t0 <= t <= h.t1]
        if not inside:
            return h
        n = len(inside)
        sxx = sum(c[0][0] for c in inside) / n + h.spread[0] ** 2
        syy = sum(c[1][1] for c in inside) / n + h.spread[1] ** 2
        sxy = sum(c[0][1] for c in inside) / n
        return replace(h, cov_mm2=((sxx, sxy), (sxy, syy)))


def track(samples: Iterable[Sample], profile: Profile) -> tuple[list[Hold], list[SpanHold]]:
    """Holds and spans from a list of samples, by a fresh `Tracker` (deterministic).

    Samples are used in time order (a stable sort, so equal times keep their order)."""
    tr = Tracker(profile)
    for s in sorted(samples, key=lambda s: s.t):
        tr.update(s)
    return tr.holds(), tr.spans()
