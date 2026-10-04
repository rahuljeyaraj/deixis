"""One list of scenarios, three robots: each answer right in its own frame and units (8.3).

The same core code, unchanged, serves the plotter (paper, point), the vacuum (floor, a 1 m
box) and the arm (table, a box the size of the moved block). Only the profile differs.
"""

import pytest

from deixis.where import Where
from support import PROFILE_NAMES, call, hold, profile, run, say, scale, thing

BLOCK = thing("red", (100, 100, 50, 40), label="red block", height_mm=40)


def here_call(name, place):
    return {
        "plotter-a5": call("place_drawing", object="star", place=place),
        "vacuum-floor": call("clean", place=place),
        "arm-table": call("move", object={"label": "red block"}, place=place),
    }[name]


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_here_plus_a_hold(name):
    p = profile(name)
    x, y = scale(p, 0.6, 0.5)
    r = run(
        p,
        words=say("put it here"),
        holds=[hold(10.2, 10.8, x, y)],
        things=[BLOCK] if name == "arm-table" else [],
        calls=[here_call(name, {"deixis": "here", "word_index": 2})],
    )
    w = r.calls[0].where("place")
    assert w.frame == p.frame and w.mean_mm == (x, y)
    if name == "plotter-a5":
        assert w.kind == "point"
    elif name == "vacuum-floor":
        assert w.kind == "box" and w.box_mm == (x - 500, y - 500, 1000, 1000)
    else:
        assert w.kind == "box" and w.box_mm == (x - 25, y - 20, 50, 40)  # the block's size


@pytest.mark.parametrize("name", ["plotter-a5", "vacuum-floor"])
def test_from_here_to_here(name):
    p = profile(name)
    (ax, ay), (bx, by) = scale(p, 0.2, 0.3), scale(p, 0.7, 0.8)
    r = run(
        p,
        words=say("from here and to here", 10.0, 0.6),
        holds=[hold(10.3, 11.1, ax, ay), hold(12.1, 12.9, bx, by)],
        calls=[here_call(name, {"from": {"deixis": "here", "word_index": 1},
                                "to": {"deixis": "here", "word_index": 4}})],
    )  # fmt: skip
    w = r.calls[0].where("place")
    if name == "plotter-a5":  # its place wants a point: the middle of the two
        assert w.kind == "point" and w.mean_mm == pytest.approx(((ax + bx) / 2, (ay + by) / 2))
    else:
        assert w.kind == "box" and w.box_mm == pytest.approx((ax, ay, bx - ax, by - ay))


@pytest.mark.parametrize("name", ["plotter-a5", "arm-table"])
def test_that_plus_a_hold_near_a_thing(name):
    p = profile(name)
    target = {"deixis": "that", "word_index": 1}
    if name == "arm-table":
        c = call("move", object=target, place={"deixis": "here", "word_index": 3})
    else:
        c = call("place_drawing", object="hat", place={"relation": "on", "target": target})
    r = run(
        p,
        words=say("put that over here"),
        holds=[hold(10.0, 10.6, 120, 118), hold(10.8, 11.4, 400, 150)][
            : 2 if c["name"] == "move" else 1
        ],
        things=[BLOCK],
        calls=[c],
    )
    if name == "arm-table":
        assert r.calls[0].where("object").thing == "red"
    else:
        assert r.calls[0].where("place").evidence["target"] == "red"


@pytest.mark.parametrize("name,step", [("plotter-a5", 5), ("vacuum-floor", 250), ("arm-table", 30)])
def test_a_bit_left(name, step):
    p = profile(name)
    x, y = scale(p, 0.5, 0.5)
    prev = Where(frame=p.frame, kind="point", mean_mm=(x, y), source="gesture")
    r = run(p, words=say("a bit left"), calls=[call("adjust", dx=-1)],
            context={"previous": {"place": prev.model_dump(mode="json")}})  # fmt: skip
    assert r.calls[0].where("place").mean_mm == pytest.approx((x - step, y))
