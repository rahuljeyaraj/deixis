"""The 11 kinds of deixis (plan 8.1), each on every profile it applies to (C12).

Every request goes through `support.run`, which also checks that the same request gives
byte-identical JSON twice.
"""

import math

import pytest

from deixis.where import Where
from support import (
    PROFILE_NAMES,
    call,
    hold,
    profile,
    run,
    say,
    scale,
    thing,
    with_actions,
)

# --------------------------------------------------------------------------- helpers

RED = thing("red", (20, 20, 50, 50), label="red block", attrs={"color": "red"}, height_mm=50, t=1)


def place_call(name, place):
    """One call with one place argument, for each robot."""
    if name == "plotter-a5":
        return call("place_drawing", object="cat", **({"place": place} if place else {}))
    if name == "vacuum-floor":
        return call("clean", place=place or {"pointed": True})
    return call("move", object={"label": "red block"}, place=place or {"pointed": True})


def scene_for(name):
    return [RED] if name == "arm-table" else []


def place_of(r):
    assert r.status == "ok", r.ask
    return r.calls[0].where("place")


HERE = {"deixis": "here", "word_index": 2}  # "put it here": "here" is at 10.6..10.9 s


def put_here(name, *holds, **kw):
    p = profile(name)
    return run(
        p,
        words=say("put it here"),
        holds=list(holds),
        things=scene_for(name),
        calls=[place_call(name, HERE)],
        **kw,
    )


# --------------------------------------------------------------------------- kind 1: place


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind1_here_with_one_hold(name):
    p = profile(name)
    x, y = scale(p, 0.4, 0.4)
    w = place_of(put_here(name, hold(10.2, 10.7, x, y)))
    assert w.mean_mm == pytest.approx((x, y), abs=0.5) and w.source == "gesture"
    assert w.frame == p.frame and w.evidence["word"] == "here"
    assert w.kind == ("point" if name == "plotter-a5" else "box")


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind1_the_timing_window(name):
    x, y = scale(profile(name), 0.4, 0.4)
    assert put_here(name, hold(8.6, 9.2, x, y)).status == "ok"  # ended 1.4 s before "here"
    late = put_here(name, hold(8.4, 9.0, x, y))  # ended 1.6 s before
    assert late.status == "ask" and late.ask.reason == "no_gesture"
    assert late.ask.word.text == "here" and (late.ask.call, late.ask.arg) == (0, "place")
    after = put_here(name, hold(11.0, 11.3, x, y))  # starts just after the word
    assert place_of(after).evidence["lead_s"] < 0


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind1_equally_near_holds_in_one_place_the_earlier_wins(name):
    p = profile(name)
    x, y = scale(p, 0.4, 0.4)
    d = p.pointer.same_place_mm / 5
    early, late = hold(10.15, 10.55, x, y), hold(10.95, 11.35, x + d, y)
    w = place_of(put_here(name, late, early))
    assert w.evidence["hold_t"] == [10.15, 10.55]


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind1_unsteady_and_ambiguous_ask(name):
    p = profile(name)
    x, y = scale(p, 0.4, 0.4)
    wobbly = put_here(name, hold(10.2, 10.7, x, y, spread=2 * p.pointer.max_spread_mm))
    assert wobbly.ask.reason == "unsteady" and len(wobbly.ask.candidates) == 1
    x2, y2 = scale(p, 0.8, 0.7)
    two = put_here(name, hold(10.15, 10.55, x, y), hold(10.95, 11.35, x2, y2))
    assert two.ask.reason == "ambiguous" and len(two.ask.candidates) == 2
    assert two.ask.question == p.ask_text("ambiguous")


# --------------------------------------------------------------------------- kind 2: no word


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind2_no_pointing_word_takes_the_last_hold(name):
    p = profile(name)
    x, y = scale(p, 0.3, 0.6)
    x2, y2 = scale(p, 0.7, 0.2)
    req = {
        "words": say("a tree please"),
        "things": scene_for(name),
        "calls": [place_call(name, None)],
    }
    r = run(p, holds=[hold(8.5, 9.0, x2, y2), hold(9.6, 10.4, x, y)], **req)
    w = place_of(r)
    assert w.mean_mm == pytest.approx((x, y), abs=0.5) and w.evidence["rule"] == "last_hold"
    none = run(p, holds=[], **req)
    assert none.ask.reason == "no_gesture" and none.ask.question == p.ask_text(
        "no_gesture", "pointed"
    )


def test_kind2_a_pinch_is_not_the_place(plotter):
    req = {
        "words": say("a tree please"),
        "holds": [hold(9.0, 9.5, 50, 50), hold(10.0, 10.6, 120, 120)],
        "spans": [{"t0": 10.0, "t1": 10.6, "span_mm": 30, "kind": "pinch", "center": [120, 120]}],
        "calls": [call("place_drawing", object="tree")],
    }
    assert place_of(run(plotter, **req)).mean_mm == (50, 50)


# --------------------------------------------------------------------------- kind 3: object

HOUSE = thing("house", (40, 40, 40, 30), t=1)
TREE = thing("tree", (120, 40, 20, 50), t=2)


def object_request(name, obj, things, holds):
    p = profile(name)
    if name == "arm-table":
        c = call("move", object=obj, place={"deixis": "here", "word_index": 4})
    else:
        c = call("place_drawing", object="eye", place={"relation": "next_to", "target": obj})
    return run(p, words=say("put that one over here"), holds=holds, things=things, calls=[c])


def object_of(r, name):
    assert r.status == "ok", r.ask
    if name == "arm-table":
        return r.calls[0].where("object").thing
    return r.calls[0].where("place").evidence["target"]


THAT = {"deixis": "that", "word_index": 1}  # at 10.3..10.6
HERE_HOLD = hold(11.0, 11.5, 300, 300)


@pytest.mark.parametrize("name", ["plotter-a5", "arm-table"])
def test_kind3_that_one(name):
    near_house = object_request(name, THAT, [HOUSE, TREE], [hold(10.1, 10.6, 60, 55), HERE_HOLD])
    assert object_of(near_house, name) == "house"
    between = object_request(name, THAT, [HOUSE, TREE], [hold(10.1, 10.6, 100, 55), HERE_HOLD])
    assert between.ask.reason == "ambiguous_object"
    assert {c.thing for c in between.ask.candidates} == {"house", "tree"}
    assert set(between.ask.options) == {"house", "tree"}
    nothing = object_request(name, THAT, [HOUSE, TREE], [hold(10.1, 10.6, 60, 190), HERE_HOLD])
    assert nothing.ask.reason == "no_object"


@pytest.mark.parametrize("name", ["plotter-a5", "arm-table"])
def test_kind3_that_cat_picks_the_cat(name):
    cat = thing("cat", (100, 40, 15, 15), label="cat")
    house = thing("house", (118, 40, 15, 15), label="house")
    point = [hold(10.1, 10.6, 116.5, 47), HERE_HOLD]
    assert object_request(name, THAT, [cat, house], point).ask.reason == "ambiguous_object"
    that_cat = {**THAT, "label": "cat"}
    assert object_of(object_request(name, that_cat, [cat, house], point), name) == "cat"


def test_kind3_by_attribute_and_label_ties(arm, plotter):
    blue = thing("b1", (300, 100, 50, 50), label="block", attrs={"color": "blue"}, t=2)
    red = thing("r1", (100, 100, 50, 50), label="block", attrs={"color": "red"}, t=1)
    one = object_request("arm-table", {"label": "the blue one"}, [red, blue], [HERE_HOLD])
    assert object_of(one, "arm-table") == "b1"
    blue2 = thing("b2", (400, 100, 50, 50), label="block", attrs={"color": "blue"}, t=3)
    tie = object_request("arm-table", {"label": "the blue one"}, [red, blue, blue2], [HERE_HOLD])
    assert tie.ask.reason == "ambiguous_object" and len(tie.ask.candidates) == 2
    newest = object_request("plotter-a5", {"label": "blue"}, [red, blue, blue2], [HERE_HOLD])
    assert object_of(newest, "plotter-a5") == "b2"  # plotter: most recent


def test_kind3_an_arrow_from_an_empty_spot(plotter):
    c = call("draw_connector", kind="arrow", from_=THAT, to={"deixis": "this", "word_index": 3})
    r = run(
        plotter,
        words=say("from that to this"),
        holds=[hold(10.1, 10.6, 60, 55), hold(10.8, 11.3, 150, 150)],
        things=[HOUSE],
        calls=[c],
    )
    frm, to = r.calls[0].where("from"), r.calls[0].where("to")
    assert frm.thing == "house" and to.thing is None and to.mean_mm == (150, 150)


# --------------------------------------------------------------------------- kind 4: size by hand

THIS_BIG = {"word_index": 2}  # "a cat this big": "this" at 10.6..10.9


def size_request(p, spans, size=THIS_BIG, action="place_drawing", holds=None):
    return run(
        p,
        words=say("a cat this big"),
        holds=holds if holds is not None else [hold(9.5, 10.0, 100, 100)],
        spans=spans,
        calls=[call(action, object="cat", size=size)],
    )


def span(t0, t1, mm, kind="pinch"):
    return {"t0": t0, "t1": t1, "span_mm": mm, "kind": kind, "center": [100, 100]}


def test_kind4_size_by_hand(plotter):
    pinch = size_request(plotter, [span(10.4, 11.2, 52)]).calls[0].size("size")
    assert pinch.mm == 52 and pinch.source == "gesture" and pinch.evidence["size_kind"] == "pinch"
    hands = size_request(plotter, [span(10.4, 11.2, 150, "two_hands")]).calls[0].size("size")
    assert hands.mm == 150 and hands.evidence["size_kind"] == "two_hands"
    none = size_request(plotter, [])
    assert none.ask.reason == "no_size" and none.ask.question.startswith("How big?")
    huge = size_request(plotter, [span(10.4, 11.2, 260, "two_hands")])
    assert huge.ask.reason == "no_size"  # above the profile's 200 mm


def test_kind4_measure_an_answer(plotter):
    """ "How big?" ... "this big": the hidden `measure` action sizes the answer alone."""
    words = say("this big")  # "this" at 10.0..10.3
    r = run(plotter, words=words, spans=[span(9.8, 10.9, 47)],
            calls=[call("measure", size={"word_index": 0})])  # fmt: skip
    s = r.calls[0].size("size")
    assert s.mm == 47 and s.evidence["span_t"] == [9.8, 10.9]
    r = run(plotter, words=words, spans=[], calls=[call("measure", size={"word_index": 0})])
    assert r.ask.reason == "no_size"


def test_kind4_no_size_gestures_on_this_robot(vacuum):
    spot = {"name": "spot", "params": [{"name": "place", "type": "place", "want": "box"},
                                       {"name": "size", "type": "size"}]}  # fmt: skip
    p = with_actions(vacuum, spot)
    r = run(p, words=say("a spot this big"), calls=[call("spot", size=THIS_BIG)])
    assert r.ask.reason == "unsupported" and r.ask.arg == "size"


# --------------------------------------------------------------------------- kind 5: size by word


@pytest.mark.parametrize("word,mm", [("tiny", 20), ("small", 35), ("medium", 55), ("big", 80),
                                     ("large", 80), ("whole page", 0.95 * 196)])  # fmt: skip
def test_kind5_presets(plotter, word, mm):
    s = size_request(plotter, [], size=word).calls[0].size("size")
    assert s.mm == pytest.approx(mm) and s.source == "verbal_preset"


def test_kind5_same_size_as_that(plotter, vacuum):
    house = thing("house", (40, 40, 60, 30))
    r = run(
        plotter,
        words=say("a cat as big as that"),
        holds=[hold(11.0, 11.6, 60, 50)],
        things=[house],
        calls=[call("place_drawing", object="cat",
                    size={"same_as": {"deixis": "that", "word_index": 5}})],
    )  # fmt: skip
    s = r.calls[0].size("size")
    assert s.mm == 60 and s.source == "object_ref" and s.evidence["same_as"] == "house"
    spot = {"name": "spot", "params": [{"name": "place", "type": "place", "want": "box"},
                                       {"name": "size", "type": "size"}]}  # fmt: skip
    p = with_actions(vacuum, spot)
    r = run(
        p,
        words=say("a big spot here"),
        holds=[hold(10.5, 11.0, 1500, 1200, 20)],
        calls=[call("spot", size="big", place={"deixis": "here", "word_index": 3})],
    )
    w = r.calls[0].where("place")
    assert w.box_mm == (750, 450, 1500, 1500) and w.evidence["box_from"] == "size"


# --------------------------------------------------------------------------- kind 6: two places

FROM_TO = {"from": {"deixis": "here", "word_index": 2}, "to": {"deixis": "here", "word_index": 5}}
SIX = say("clean from here and to here", 10.0, 0.6)  # "here" at 11.2..11.8 and 13.0..13.6


def test_kind6_two_places_make_a_box(vacuum):
    a, b = hold(10.8, 11.6, 2100, 1500, 20), hold(12.8, 13.7, 800, 600, 20)
    w = place_of(run(vacuum, words=SIX, holds=[b, a], calls=[call("clean", place=FROM_TO)]))
    assert w.kind == "box" and w.box_mm == (800, 600, 1300, 900)
    assert w.evidence["corners"] == [[2100, 1500], [800, 600]]  # in time order


def test_kind6_a_path_when_the_robot_wants_one(vacuum):
    route = {"name": "route", "params": [{"name": "place", "type": "place", "want": "path"}]}
    p = with_actions(vacuum, route)
    a, b = hold(10.8, 11.6, 2100, 1500, 20), hold(12.8, 13.7, 800, 600, 20)
    w = place_of(run(p, words=SIX, holds=[a, b], calls=[call("route", place=FROM_TO)]))
    assert w.kind == "path" and w.path_mm == [(2100, 1500), (800, 600)]
    point = run(
        p, words=SIX, holds=[a], calls=[call("route", place={"deixis": "here", "word_index": 2})]
    )
    assert point.ask.reason == "unsupported"


def test_kind6_only_one_hold_asks(vacuum):
    r = run(vacuum, words=SIX, holds=[hold(10.8, 11.6, 2100, 1500, 20)],
            calls=[call("clean", place=FROM_TO)])  # fmt: skip
    assert r.ask.reason == "no_gesture" and r.ask.word.t0 == 13.0


# --------------------------------------------------------------------------- kind 7: relations

TARGET = thing("t", (100, 100, 40, 40), height_mm=30)


def relation_request(name, rel, things=(TARGET,), **extra):
    p = profile(name)
    place = {"relation": rel, "target": {"label": "t"}, **extra}
    if name == "arm-table":
        c = call("move", object={"label": "red block"}, place=place)
        things = (*things, RED)
    elif name == "vacuum-floor":
        c = call("clean", place=place)
    else:
        c = call("place_drawing", object="cat", place=place, size="tiny")
    if name == "vacuum-floor":
        things = [
            thing(o["id"], tuple(v * 10 for v in o["box_mm"]), height_mm=o.get("height_mm"))
            for o in things
        ]
    return run(p, words=say("put it next to that"), things=list(things), calls=[c])


EXPECT = {
    # relation: (page, surface, ground); "u" = unsupported, "z" = stacked
    "left_of": ("x<", "x<", "x<"),
    "right_of": ("x>", "x>", "x>"),
    "above": ("y>", "z", "u"),
    "below": ("y<", "u", "u"),
    "behind": ("y>", "y>", "y>"),
    "in_front_of": ("y<", "y<", "y<"),
    "on": ("y~", "z", "z"),
    "inside": ("c", "c", "c"),
    "next_to": ("x>", "x>", "x>"),
}


VIEW_COLUMN = {"plotter-a5": 0, "arm-table": 1, "vacuum-floor": 2}  # page, surface, ground


@pytest.mark.parametrize("rel", sorted(EXPECT))
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_kind7_relations_by_view(rel, name):
    r = relation_request(name, rel)
    want = EXPECT[rel][VIEW_COLUMN[name]]
    if want == "u":
        assert r.ask.reason == "unsupported" and r.ask.arg == "place"
        return
    w = place_of(r)
    k = 10 if name == "vacuum-floor" else 1
    x0, y0, x1, y1 = 100 * k, 100 * k, 140 * k, 140 * k
    cx, cy = w.mean_mm
    assert w.evidence["relation"] == rel and w.evidence["target"] == "t"
    if want == "x<":
        assert cx < x0 and y0 < cy < y1
    elif want == "x>":
        assert cx > x1 and y0 < cy < y1
    elif want == "y>":
        assert cy > y1 and x0 < cx < x1
    elif want == "y<":
        assert cy < y0 and x0 < cx < x1
    elif want == "y~":
        assert y1 < cy < y1 + 20 and w.z_mm is None
    elif want == "z":
        assert (cx, cy) == ((x0 + x1) / 2, (y0 + y1) / 2) and w.z_mm == 30
    else:
        assert (cx, cy) == ((x0 + x1) / 2, (y0 + y1) / 2)


def test_kind7_between_these_two(plotter):
    a, b = thing("a", (20, 20, 20, 20)), thing("b", (120, 60, 20, 20))
    place = {"relation": "between", "target": {"deixis": "these", "word_index": 3, "count": 2}}
    r = run(
        plotter,
        words=say("a bird between these two"),
        holds=[hold(10.3, 10.8, 30, 30), hold(11.0, 11.5, 130, 70)],
        things=[a, b],
        calls=[call("place_drawing", object="bird", place=place)],
    )
    w = place_of(r)
    assert w.mean_mm == (80, 50) and (w.evidence["target"], w.evidence["target2"]) == ("a", "b")
    one = run(plotter, words=say("a bird between these two"), holds=[hold(10.3, 10.8, 30, 30)],
              things=[a, b], calls=[call("place_drawing", object="bird", place=place)])  # fmt: skip
    assert one.ask.reason == "no_gesture"


def test_kind7_inside_shrinks_the_size(plotter):
    box = thing("box", (50, 50, 60, 40))
    r = run(plotter, words=say("a hand inside the box"), things=[box],
            calls=[call("place_drawing", object="hand",
                        place={"relation": "inside", "target": {"label": "box"}})])  # fmt: skip
    w = place_of(r)
    assert w.mean_mm == (80, 70) and w.evidence["size_mm"] == pytest.approx(0.7 * 40)


# --------------------------------------------------------------------------- kind 8: last thing


def test_kind8_it_is_the_newest_selectable_thing(arm):
    old = thing("old", (100, 100, 50, 50), t=1)
    new = thing("new", (300, 100, 50, 50), t=5)
    note = thing("note", (400, 300, 50, 10), t=9, selectable=False)
    req = {
        "words": say("move it here"),
        "holds": [hold(10.3, 10.9, 500, 300)],
        "calls": [call("move", object={"last": True}, place={"deixis": "here", "word_index": 2})],
    }
    assert run(arm, things=[old, new, note], **req).calls[0].where("object").thing == "new"
    known = run(arm, things=[old, new, note], context={"last_thing": "old"}, **req)
    assert known.calls[0].where("object").thing == "old"
    empty = run(arm, things=[note], **req)
    assert empty.ask.reason == "no_object" and empty.ask.question == arm.ask_text(
        "no_object", "empty"
    )


# --------------------------------------------------------------------------- kind 9: small fixes


@pytest.mark.parametrize("name,step", [("plotter-a5", 5), ("vacuum-floor", 250), ("arm-table", 30)])
def test_kind9_a_bit_left(name, step):
    p = profile(name)
    x, y = scale(p, 0.5, 0.5)
    prev = Where(frame=p.frame, kind="box", box_mm=(x - 20, y - 10, 40, 20), source="gesture")
    r = run(p, words=say("a bit left"), calls=[call("adjust", dx=-1)],
            context={"previous": {"place": prev.model_dump(mode="json")}})  # fmt: skip
    w = r.calls[0].where("place")
    assert w.mean_mm == pytest.approx((x - step, y)) and w.source == "adjust"
    assert w.evidence["offset_mm"] == [-step, 0]  # a robot moves its own copy by this
    bigger = run(p, words=say("bigger"), calls=[call("adjust", scale=1.5)],
                 context={"previous": {"place": prev.model_dump(mode="json"),
                                       "size": {"mm": 40, "source": "gesture"}}})  # fmt: skip
    b = bigger.calls[0]
    assert b.where("place").box_mm == pytest.approx((x - 30, y - 15, 60, 30))
    assert b.size("size").mm == 60
    nothing = run(p, words=say("a bit left"), calls=[call("adjust", dx=-1)])
    assert nothing.ask.reason == "no_previous"


# --------------------------------------------------------------------------- kind 10: unsure


def _asks():
    """A request for every Ask reason: (reason, profile name, request)."""
    p_here = [place_call("plotter-a5", HERE)]
    cases = {
        "no_gesture": ("plotter-a5", {"words": say("put it here"), "calls": p_here}),
        "ambiguous": ("plotter-a5", {"words": say("put it here"), "calls": p_here,
                                     "holds": [hold(10.15, 10.55, 30, 30),
                                               hold(10.95, 11.35, 150, 150)]}),
        "unsteady": ("plotter-a5", {"words": say("put it here"), "calls": p_here,
                                    "holds": [hold(10.2, 10.7, 30, 30, spread=9)]}),
        "no_size": ("plotter-a5", {"words": say("a cat this big"),
                                   "calls": [call("place_drawing", object="cat", size=THIS_BIG)]}),
        "no_object": ("arm-table", {"words": say("move it here"),
                                    "calls": [call("move", object={"last": True},
                                                   place={"pointed": True})]}),
        "ambiguous_object": ("arm-table", {"words": say("put that one over here"),
                                           "things": [HOUSE, TREE],
                                           "holds": [hold(10.1, 10.6, 100, 55), HERE_HOLD],
                                           "calls": [call("move", object=THAT,
                                                          place={"pointed": True})]}),
        "unknown_label": ("arm-table", {"words": say("move the fish"), "things": [RED],
                                        "calls": [call("move", object={"label": "fish"},
                                                       place={"pointed": True})]}),
        "unknown_place": ("plotter-a5", {"words": say("a cat on the roof"),
                                         "calls": [call("place_drawing", object="cat",
                                                        place={"named": "on the roof"})]}),
        "no_room": ("plotter-a5", {"words": say("a cat anywhere"),
                                   "things": [thing("full", (-5, -5, 300, 300))],
                                   "calls": [call("place_drawing", object="cat",
                                                  place={"empty_space": True})]}),
        "outside_area": ("arm-table", {"words": say("put it here"), "things": [RED],
                                       "holds": [hold(10.2, 10.7, 900, 100)],
                                       "calls": [place_call("arm-table", HERE)]}),
        "no_previous": ("arm-table", {"words": say("a bit left"),
                                      "calls": [call("adjust", dx=-1)]}),
        "unsupported": ("vacuum-floor", {"words": say("clean above that"),
                                         "things": [thing("sofa", (100, 100, 900, 400))],
                                         "calls": [call("clean", place={
                                             "relation": "above", "target": {"label": "sofa"}})]}),
    }  # fmt: skip
    return [(reason, name, req) for reason, (name, req) in cases.items()]


@pytest.mark.parametrize("reason,name,req", _asks(), ids=[a[0] for a in _asks()])
def test_kind10_every_ask_reason(reason, name, req):
    p = profile(name)
    r = run(p, **req)
    assert r.status == "ask" and r.ask.reason == reason and r.calls == []
    assert r.ask.question and "{" not in r.ask.question
    assert r.ask.call == 0 and r.ask.arg


def test_kind10_every_reason_has_a_case():
    from typing import get_args

    from deixis.where import AskReason

    assert {a[0] for a in _asks()} == set(get_args(AskReason))


def test_kind10_answers_give_ok(arm):
    req = {
        "words": say("put that one over here"),
        "things": [HOUSE, TREE],
        "holds": [hold(10.1, 10.6, 100, 55), HERE_HOLD],
        "calls": [call("move", object=THAT, place={"deixis": "here", "word_index": 4})],
    }
    ask = run(arm, **req).ask
    pick = ask.candidates[0].model_dump(mode="json")
    by_candidate = run(arm, **req, answers=[{"arg": "object", "candidate": 0, "where": pick}])
    assert by_candidate.status == "ok"
    assert by_candidate.calls[0].where("object").thing == ask.candidates[0].thing
    by_thing = run(arm, **req, answers=[{"arg": "object", "thing": "tree"}])
    assert by_thing.calls[0].where("object").thing == "tree"
    again = hold(12.0, 12.5, 60, 55)  # pointed again, at the house
    by_hold = run(arm, **req, answers=[{"arg": "object", "hold": again}])
    assert by_hold.calls[0].where("object").thing == "house"


def test_kind10_partial_round_trips(arm):
    req = {
        "words": say("put that one over here"),
        "things": [HOUSE, TREE],
        "holds": [hold(10.1, 10.6, 60, 55)],
        "calls": [call("move", object=THAT, place={"deixis": "here", "word_index": 4})],
    }
    first = run(arm, **req)
    assert first.ask.arg == "place" and [a.arg for a in first.partial] == ["object"]
    back = [a.model_dump(mode="json") for a in first.partial]
    back.append({"arg": "place", "hold": hold(13.0, 13.5, 300, 200)})
    second = run(arm, **req, answers=back)
    assert second.status == "ok"
    assert second.calls[0].where("object") == first.partial[0].where


# --------------------------------------------------------------------------- kind 11: standing spot

FPS = 30


def feet(t0, t1, at, sway=0.0, hz=0.5):
    n = int(round((t1 - t0) * FPS))
    return [
        {"t": round(t0 + k / FPS, 4),
         "point": [at[0] + sway * math.sin(2 * math.pi * hz * k / FPS), at[1]]}
        for k in range(n)
    ]  # fmt: skip


def walk(t0, t1, a, b):
    n = int(round((t1 - t0) * FPS))
    return [{"t": round(t0 + k / FPS, 4), "point": [a[0] + (b[0] - a[0]) * k / n,
                                                     a[1] + (b[1] - a[1]) * k / n]}
            for k in range(n)]  # fmt: skip


CLEAN_HERE = {
    "words": say("clean here", 10.0),
    "calls": [call("clean", place={"deixis": "here", "word_index": 1})],
}


def test_kind11_standing_one_second_is_a_box(vacuum):
    samples = walk(7, 9.3, (0, 0), (1520, 980)) + feet(9.3, 10.6, (1520, 980))
    samples += walk(10.6, 12, (1520, 980), (2900, 2400))
    w = place_of(run(vacuum, samples=samples, **CLEAN_HERE))
    assert w.kind == "box" and w.box_mm[2:] == (1000, 1000)
    assert w.mean_mm == pytest.approx((1520, 980), abs=40) and w.evidence["box_from"] == "default"


def test_kind11_swaying_holds_walking_does_not(vacuum):
    sway = walk(7, 9.0, (0, 0), (1500, 1000)) + feet(9.0, 11.0, (1500, 1000), sway=60)
    assert run(vacuum, samples=sway, **CLEAN_HERE).status == "ok"
    walking = walk(7, 12, (0, 0), (2500, 2000))
    assert run(vacuum, samples=walking, **CLEAN_HERE).ask.reason == "no_gesture"


# --------------------------------------------------------------------------- the rest of 8.1


def test_outside_the_area_is_flagged_or_asked_never_moved(plotter, arm):
    far = hold(10.2, 10.7, 260, 50)  # right of the plotter's area
    w = place_of(put_here("plotter-a5", far))
    assert w.mean_mm == (260, 50) and w.evidence["outside"] is True
    inside = place_of(put_here("plotter-a5", hold(10.2, 10.7, 100, 50)))
    assert "outside" not in inside.evidence
    asked = put_here("arm-table", hold(10.2, 10.7, 700, 100))
    assert asked.ask.reason == "outside_area"
    assert asked.ask.candidates[0].mean_mm == (700, 100)  # the place, not moved


def test_a_house_here_and_a_tree_there_use_two_holds(plotter):
    r = run(
        plotter,
        words=say("a house here and a tree there"),
        holds=[hold(10.3, 10.9, 40, 40), hold(11.5, 12.1, 160, 120)],
        calls=[call("place_drawing", object="house", place={"deixis": "here", "word_index": 2}),
               call("place_drawing", object="tree", place={"deixis": "there", "word_index": 6})],
    )  # fmt: skip
    a, b = (c.where("place").mean_mm for c in r.calls)
    assert (a, b) == ((40, 40), (160, 120)) and len(r.holds) == 2


def test_clicks_are_holds(arm):
    click = hold(10.7, 10.7, 480, 90, spread=0.0, source="click")
    w = place_of(put_here("arm-table", click))
    assert w.mean_mm == (480, 90) and w.box_mm == (455, 65, 50, 50)
    assert w.cov_mm2 == ((25, 0), (0, 25))  # the base sigma only


def test_a_holds_own_covariance_is_used(arm):
    gaze = hold(10.2, 10.7, 300, 200, cov_mm2=[[400, 0], [0, 900]], source="gaze")
    assert place_of(put_here("arm-table", gaze)).cov_mm2 == ((400, 0), (0, 900))


def test_only_one_call_is_resolved_but_all_words_count(plotter):
    req = {
        "words": say("a house here and a tree there"),
        "holds": [hold(10.3, 10.9, 40, 40), hold(11.5, 12.1, 160, 120)],
        "calls": [
            call("place_drawing", object="house", place={"deixis": "here", "word_index": 2}),
            call("place_drawing", object="tree", place={"deixis": "there", "word_index": 6}),
        ],
    }
    r = run(plotter, only=1, **req)
    assert [c.index for c in r.calls] == [1] and r.calls[0].where("place").mean_mm == (160, 120)
