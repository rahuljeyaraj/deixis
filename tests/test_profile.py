"""Profiles load, and the timing, hold and span parameters are built from them (C4)."""

import pytest
from pydantic import ValidationError

from deixis.holds import HoldParams
from deixis.profile import Profile, load_profile
from deixis.resolver import ResolverParams, pick_hold
from deixis.types import Word
from support import PROFILE_NAMES, profile


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_example_profiles_load(name):
    p = profile(name)
    assert p.name == name and p.version == "1.0"
    assert p.area.w > 0 and p.area.h > 0
    assert {a.name for a in p.all_actions()} >= {"adjust", "confirm", "cancel", "stop"}
    assert load_profile(p.model_dump(mode="json")) == p  # round trip through JSON


def test_plotter_params_match_todays_plotter(plotter):
    """The plotter's profile gives exactly the parameters the plotter used before the split."""
    r = plotter.resolver_params()
    assert (r.before_s, r.after_s, r.no_word_before_s, r.time_sigma_s) == (1.5, 0.5, 2.0, 0.6)
    assert (r.ambiguity_ratio, r.same_place_mm, r.base_sigma_mm, r.max_spread_mm) == (
        2.0, 10.0, 1.5, 6.0
    )  # fmt: skip
    assert r.size_overlap == 0.5
    assert ResolverParams(asks=r.asks) == r
    h = plotter.hold_params()
    assert h == HoldParams(history_s=120.0, frame="paper", source="finger")
    s = plotter.span_params()
    assert s is not None and (s.max_rate_mm_s, s.min_duration_s, s.min_span_mm) == (25, 0.2, 8)
    f = plotter.filter_params()
    assert f is not None and (f.min_cutoff, f.beta, f.d_cutoff) == (1.0, 0.02, 1.0)


def test_vacuum_has_no_size_gestures_and_slow_feet(vacuum):
    assert vacuum.span_params() is None
    h = vacuum.hold_params()
    assert h.frame == "floor" and h.source == "feet" and h.speed_enter_mm_s == 120
    assert vacuum.sizes.default_mm == 1000 and vacuum.view == "ground"


def test_ask_texts_come_from_the_profile(plotter, arm):
    word = Word("here", 5.0, 5.3)
    ask = pick_hold(word, [], plotter.resolver_params())
    assert ask.question == "Where do you mean? Point at the paper."
    assert pick_hold(word, [], arm.resolver_params()).question == "Where? Point at the spot."
    assert pick_hold(word, []).question == "Where do you mean? Please point."  # the default
    assert arm.ask_text("ambiguous_object", options="red block or the blue block") == (
        "This one or that one?"
    )
    assert plotter.ask_text("ambiguous_object", options="house or the tree") == (
        "The house or the tree?"
    )
    assert plotter.ask_text("no_object", "empty").startswith("There's nothing drawn yet")
    assert plotter.ask_text("unsupported") == "I can't do that yet."  # the default


def _base(**kw):
    d = {"name": "x", "frame": "f", "area_mm": [0, 0, 10, 10]}
    d.update(kw)
    return d


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"viewer": {"right": [0.7, 0.7]}}, "unit vectors"),
        ({"viewer": {"right": [1, 0], "away": [-1, 0]}}, "perpendicular"),
        ({"area_mm": [0, 0, 0, 10]}, "positive"),
        ({"view": "sky"}, "view"),
        ({"asks": {"why_not": "?"}}, "unknown ask"),
        ({"version": "2.0"}, "not 1.x"),
        ({"sizes": {"presets": {"big": 80}, "synonyms": {"huge": "giant"}}}, "unknown presets"),
        ({"actions": [{"name": "stop"}]}, "standard"),
        ({"actions": [{"name": "go"}, {"name": "go"}]}, "unique"),
        ({"words": {"demonstratives": ["ici"]}}, "unknown demonstratives"),
        ({"colour": "red"}, "Extra inputs"),
    ],
)
def test_bad_profiles_are_refused(bad, match):
    with pytest.raises(ValidationError, match=match):
        Profile.model_validate(_base(**bad))


def test_minimal_profile_has_defaults():
    p = Profile.model_validate(_base())
    assert p.view == "page" and p.spans is None and p.outside == "keep"
    assert p.resolver_params() == ResolverParams()
