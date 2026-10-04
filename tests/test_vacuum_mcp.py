"""The vacuum on the core over MCP (C17, plan U2): feet in, a cleaning box on the robot's
map out, with no change to the core.

The vacuum app sends the feet's midpoint on the floor (one sample per camera frame, mm in
the floor frame of its 4 floor markers) and the words. "Clean here" is a 1 m box around
where the person stood; "clean from here ... to here" takes the two standing spots as
corners. The app then maps the box from the floor to the robot's own map with a similarity
fitted from a few points both know (the dock, the robot's lid), all over MCP.
"""

import math

import numpy as np
import pytest

from support import call, profile, say

pytest.importorskip("mcp")

from deixis.api import resolve  # noqa: E402
from deixis.mcp import DeixisClient  # noqa: E402

FPS = 30
JITTER_MM = 20  # per-frame noise on the feet (holds survive about 30 mm, 4 Oct)
HERE = {"deixis": "here", "word_index": 1}  # "clean here"


@pytest.fixture(scope="module")
def vacuum_client():
    c = DeixisClient(profile("vacuum-floor"), mode="mcp", fallback=None)
    yield c
    c.close()


class Feet:
    """A person on the floor, one sample per frame, with seeded noise."""

    def __init__(self, at, t=6.0, seed=0):
        self.rng = np.random.default_rng(seed)
        self.at, self.t, self.samples = at, t, []

    def _add(self, p):
        n = self.rng.normal(0, JITTER_MM, 2)
        self.samples.append({"t": round(self.t, 4), "point": [p[0] + n[0], p[1] + n[1]]})
        self.t += 1 / FPS

    def walk(self, to, speed_mm_s=1000):
        d = math.dist(self.at, to)
        n = max(1, round(d / speed_mm_s * FPS))
        a = self.at
        for k in range(n):
            f = k / n
            self._add((a[0] + (to[0] - a[0]) * f, a[1] + (to[1] - a[1]) * f))
        self.at = to
        return self

    def stand(self, s):
        for _ in range(round(s * FPS)):
            self._add(self.at)
        return self


def ask(client, words, samples, place):
    req = {"words": words, "samples": samples, "calls": [call("clean", place=place)]}
    r = client.resolve(req)
    assert r.model_dump_json() == resolve(req, profile("vacuum-floor")).model_dump_json()
    assert client.over_mcp
    return r


# Floor (the markers' frame) to map (the robot's own frame): the robot's map is turned
# about 30 degrees, mirrored and shifted from the floor markers, as on rahul's floor.
TURN = math.radians(30)


def to_map(p):
    x, y = p
    c, s = math.cos(TURN), math.sin(TURN)
    return (1850 + c * x + s * y, -950 + s * x - c * y)  # a turn and a mirror


def floor_to_map(client, seed=1):
    """Fit from the dock and 3 robot positions, each seen with about 5 mm of error."""
    rng = np.random.default_rng(seed)
    floor = [(200, 150), (2600, 300), (2400, 2200), (500, 2000)]
    seen = [(x + rng.normal(0, 5), y + rng.normal(0, 5)) for x, y in floor]
    return client.fit_transform(
        seen,
        [to_map(p) for p in floor],
        "similarity",
        "floor",
        "map",
        allow_mirror=True,
    )


def test_clean_here_gives_a_box_on_the_robots_map(vacuum_client):
    spot = (1520, 980)
    feet = Feet((0, 0)).walk(spot).stand(1.6).walk((2900, 2400))
    here_at = 6 + 1.82 / 1 + 0.4  # "here" while standing (walk ~1.8 s, then 0.4 s)
    words = say("clean here", here_at - 0.3)
    r = ask(vacuum_client, words, feet.samples, HERE)
    w = r.calls[0].where("place")
    assert w.frame == "floor" and w.kind == "box" and w.box_mm[2:] == (1000, 1000)
    assert math.dist(w.mean_mm, spot) < 40 and w.evidence["box_from"] == "default"

    fit = floor_to_map(vacuum_client)
    assert fit.rms_mm < 10 and fit.pairs == 4
    m = vacuum_client.apply_transform(w, fit)
    assert m.frame == "map" and m.evidence["from_frame"] == "floor"
    assert math.dist(m.mean_mm, to_map(w.mean_mm)) < 10
    # The zone order takes two opposite corners in whole map mm.
    x, y, bw, bh = m.box_mm
    zone = [round(x), round(y), round(x + bw), round(y + bh)]
    corners = [to_map((a, b)) for a in (w.box_mm[0], w.box_mm[0] + 1000)
               for b in (w.box_mm[1], w.box_mm[1] + 1000)]  # fmt: skip
    assert zone[0] == pytest.approx(min(c[0] for c in corners), abs=15)
    assert zone[3] == pytest.approx(max(c[1] for c in corners), abs=15)


def test_clean_from_here_to_here_takes_the_two_spots_as_corners(vacuum_client):
    a, b = (2100, 1500), (800, 600)
    feet = Feet((3000, 2400)).walk(a).stand(1.4).walk(b).stand(1.4).walk((0, 0))
    t_a = 6 + math.dist((3000, 2400), a) / 1000 + 0.5
    t_b = t_a - 0.5 + 1.4 + math.dist(a, b) / 1000 + 0.5
    words = [
        {"text": "clean", "t0": t_a - 0.6, "t1": t_a - 0.3},
        {"text": "from", "t0": t_a - 0.3, "t1": t_a},
        {"text": "here", "t0": t_a, "t1": t_a + 0.3},
        {"text": "to", "t0": t_b - 0.3, "t1": t_b},
        {"text": "here", "t0": t_b, "t1": t_b + 0.3},
    ]
    words = [{**w, "t0": round(w["t0"], 3), "t1": round(w["t1"], 3)} for w in words]
    place = {"from": {"deixis": "here", "word_index": 2}, "to": {"deixis": "here", "word_index": 4}}
    w = ask(vacuum_client, words, feet.samples, place).calls[0].where("place")
    assert w.kind == "box"
    (ax, ay), (bx, by) = w.evidence["corners"]
    assert math.dist((ax, ay), a) < 40 and math.dist((bx, by), b) < 40  # in time order
    x, y, bw, bh = w.box_mm
    assert (x, y, x + bw, y + bh) == pytest.approx((800, 600, 2100, 1500), abs=40)


def test_walking_through_without_stopping_asks(vacuum_client):
    feet = Feet((0, 0)).walk((3000, 2400), speed_mm_s=800)
    r = ask(vacuum_client, say("clean here", 8.0), feet.samples, HERE)
    assert r.status == "ask" and r.ask.reason in ("no_gesture", "unsteady")
    assert r.ask.question.startswith("Where")


def test_near_the_edge_nothing_moves_and_past_it_is_flagged(vacuum_client):
    """The floor area is what the camera's markers span; a box may stick out of it (the
    robot's own map knows the walls). A spot past the edge is flagged, never moved."""
    for spot, outside in (((200, 1200), None), ((-150, 1200), True)):
        feet = Feet((1500, 1200)).walk(spot).stand(1.6).walk((1500, 1200))
        here_at = 6 + math.dist((1500, 1200), spot) / 1000 + 0.4
        words = say("clean here", here_at - 0.3)
        r = ask(vacuum_client, words, feet.samples, HERE)
        w = r.calls[0].where("place")
        assert math.dist(w.mean_mm, spot) < 40 and w.box_mm[0] < 0  # not shifted to fit
        assert w.evidence.get("outside") is outside
