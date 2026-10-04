"""Tracker and track(): samples to holds and spans, per profile (C9)."""

import math

import numpy as np
import pytest

from deixis.tracker import Tracker, track
from deixis.types import Sample

FPS = 30


def still(t0, t1, p, jitter=0.0, seed=0, **kw):
    rng = np.random.default_rng(seed)
    n = int(round((t1 - t0) * FPS))
    return [
        Sample(t0 + k / FPS, (p[0] + rng.normal(0, jitter), p[1] + rng.normal(0, jitter)), **kw)
        for k in range(n)
    ]


def walk(t0, t1, a, b, **kw):
    n = int(round((t1 - t0) * FPS))
    return [
        Sample(t0 + k / FPS, (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n), **kw)
        for k in range(n)
    ]


def test_finger_hold_on_the_plotter(plotter):
    samples = walk(0, 0.5, (0, 0), (100, 80)) + still(0.5, 1.5, (100, 80), 0.3)
    samples += walk(1.5, 2.0, (100, 80), (180, 20))
    holds, spans = track(samples, plotter)
    assert len(holds) == 1 and spans == []
    h = holds[0]
    assert h.mean == pytest.approx((100, 80), abs=1.5) and h.frame == "paper"
    assert h.source == "finger" and 0.4 < h.t0 < 0.9 and 1.3 < h.t1 < 1.7


def test_standing_one_second_is_a_feet_hold(vacuum):
    """Kind 11: standing still about 1 s on the floor; swaying within 150 mm still holds."""
    samples = walk(0, 2, (0, 0), (1520, 980)) + still(2, 3.2, (1520, 980), 25)
    samples += walk(3.2, 5, (1520, 980), (2800, 2000))
    (h,), _ = track(samples, vacuum)
    assert h.mean == pytest.approx((1520, 980), abs=40) and h.source == "feet"
    assert h.frame == "floor" and h.t1 - h.t0 >= 0.6


def test_swaying_holds_but_walking_does_not(vacuum):
    t = np.arange(0, 2, 1 / FPS)
    sway = [Sample(float(x), (1000 + 60 * math.sin(2 * math.pi * 0.5 * x), 1000.0)) for x in t]
    assert len(track(sway, vacuum)[0]) == 1  # 0.5 Hz, 60 mm: about 190 mm/s at most
    slow_walk = walk(0, 4, (0, 0), (2000, 0))  # 500 mm/s
    assert track(slow_walk, vacuum)[0] == []


def test_not_pointing_or_lost_makes_no_hold(plotter):
    assert track(still(0, 1, (50, 50), pointing=False), plotter)[0] == []
    lost = [Sample(k / FPS, None) for k in range(30)]
    assert track(lost, plotter) == ([], [])


def test_spans_only_when_the_profile_allows_them(plotter, arm):
    pinch = [
        Sample(k / FPS, None, False, ((50.0, 50.0), (50.0 + 30, 50.0)), "pinch") for k in range(20)
    ]
    holds, spans = track(pinch, plotter)
    assert holds == [] and len(spans) == 1
    assert spans[0].span_mm == pytest.approx(30) and spans[0].center == (65, 50)
    assert track(pinch, arm) == ([], [])  # no "this big" on the arm


def test_samples_with_their_own_covariance(arm):
    gaze = still(0, 1, (300, 200), 2.0, cov_mm2=((400.0, 0.0), (0.0, 900.0)))
    (h,), _ = track(gaze, arm)
    assert h.cov_mm2 is not None
    assert h.cov_mm2[0][0] == pytest.approx(400 + h.spread[0] ** 2)
    assert h.cov_mm2[1][1] == pytest.approx(900 + h.spread[1] ** 2)
    (plain,), _ = track(still(0, 1, (300, 200), 2.0), arm)
    assert plain.cov_mm2 is None


def test_streaming_and_batch_agree_and_repeat(plotter):
    samples = still(0, 1, (10, 10), 0.2) + walk(1, 1.5, (10, 10), (90, 90))
    samples += still(1.5, 2.6, (90, 90), 0.2, seed=1)
    tr = Tracker(plotter)
    ended = [h for s in samples if (h := tr.update(s)[0]) is not None]
    assert ended == track(samples, plotter)[0][: len(ended)]
    assert tr.current() is not None and tr.current().mean == pytest.approx((90, 90), abs=1)
    assert track(samples, plotter) == track(list(reversed(samples)), plotter)  # sorted by t
    assert track(samples, plotter) == track(samples, plotter)
