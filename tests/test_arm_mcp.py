"""The arm web demo on the core over MCP (C18, plan U3): a fingertip seen at an angle, words,
blocks on a table; out come boxes on the table (and a height for "on that") or questions.

The web demo's server maps the webcam's fingertip pixels onto the table with a homography
fitted (over MCP) from the table's 4 corners on screen, then sends one request per sentence
with the fingertip samples, the blocks and the agent's calls. No change to the core.
"""

import math

import numpy as np
import pytest

from support import call, profile, thing

pytest.importorskip("mcp")

from deixis.api import resolve  # noqa: E402
from deixis.mcp import DeixisClient  # noqa: E402

FPS = 30
BLOCKS = [
    thing("red", (125, 225, 50, 50), label="red block", attrs={"color": "red"}, height_mm=50, t=1),
    thing("yellow", (215, 225, 50, 50), label="yellow block", attrs={"color": "yellow"},
          height_mm=50, t=2),
    thing("green", (300, 100, 50, 50), label="green block", attrs={"color": "green"},
          height_mm=50, t=3),
    thing("blue", (420, 260, 50, 50), label="blue block", attrs={"color": "blue"},
          height_mm=50, t=4),
]  # fmt: skip
# The 4 Oct spike's angled view: the 600 x 400 mm table's corners on a 1280 x 720 canvas.
TABLE_MM = [(0, 0), (600, 0), (600, 400), (0, 400)]
TABLE_PX = [(250, 690), (1030, 690), (900, 300), (380, 300)]


@pytest.fixture(scope="module")
def arm_client():
    c = DeixisClient(profile("arm-table"), mode="mcp", fallback=None)
    yield c
    c.close()


@pytest.fixture(scope="module")
def to_table(arm_client):
    """Pixels -> table mm, fitted over MCP from the table's 4 corners on screen."""
    tr = arm_client.fit_transform(TABLE_PX, TABLE_MM, "homography", "screen", "table")
    assert tr.max_mm < 1e-6
    return tr


class Finger:
    """A fingertip moving over the table, seen in pixels with 2 px of jitter."""

    def __init__(self, to_table, at=(300, 50), t=1.0, seed=0):
        self.tr, self.at, self.t = to_table, at, t
        self.rng = np.random.default_rng(seed)
        self.inv = np.linalg.inv(np.array(to_table.matrix))
        self.samples = []

    def _frame(self, p):
        v = self.inv @ np.array([p[0], p[1], 1.0])
        px = v[:2] / v[2] + self.rng.normal(0, 2.0, 2)  # what the webcam sees
        x, y = self.tr.apply([tuple(px)])[0]  # what the server sends
        self.samples.append({"t": round(self.t, 4), "point": [float(x), float(y)]})
        self.t += 1 / FPS

    def move(self, to, speed=400.0):
        n = max(1, round(math.dist(self.at, to) / speed * FPS))
        a = self.at
        for k in range(n):
            self._frame((a[0] + (to[0] - a[0]) * k / n, a[1] + (to[1] - a[1]) * k / n))
        self.at = to
        return self

    def hold(self, s):
        for _ in range(round(s * FPS)):
            self._frame(self.at)
        return self


def words(*timed):
    return [{"text": w, "t0": round(t, 3), "t1": round(t + 0.25, 3)} for w, t in timed]


def ask(client, **req):
    req = {"things": BLOCKS, **req}
    r = client.resolve(req)
    assert r.model_dump_json() == resolve(req, profile("arm-table")).model_dump_json()
    assert client.over_mcp
    return r


def test_put_that_here(arm_client, to_table):
    f = Finger(to_table).move((150, 250)).hold(0.7)
    t_that = f.t - 0.4
    f.move((480, 90)).hold(0.7)
    t_here = f.t - 0.4
    r = ask(arm_client, samples=f.samples,
            words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"deixis": "here", "word_index": 2})])  # fmt: skip
    c = r.calls[0]
    assert c.where("object").thing == "red"
    place = c.where("place")
    assert place.frame == "table" and place.box_mm[2:] == (50, 50)  # the red block's size
    assert math.dist(place.mean_mm, (480, 90)) < 6  # 2 px of jitter at the far side


def test_put_the_blue_one_next_to_that(arm_client, to_table):
    f = Finger(to_table).move((325, 125)).hold(0.7)
    t_that = f.t - 0.4
    w = words(("put", t_that - 1.5), ("the", t_that - 1.25), ("blue", t_that - 1.0),
              ("one", t_that - 0.75), ("next", t_that - 0.5), ("to", t_that - 0.25),
              ("that", t_that))  # fmt: skip
    r = ask(arm_client, samples=f.samples, words=w,
            calls=[call("move", object={"label": "blue one"},
                        place={"relation": "next_to",
                               "target": {"deixis": "that", "word_index": 6}})])  # fmt: skip
    c = r.calls[0]
    assert c.where("object").thing == "blue"
    place = c.where("place")
    assert place.evidence["target"] == "green"
    x, y, bw, bh = place.box_mm
    assert x >= 350 and (bw, bh) == (50, 50)  # to the viewer's right of the green block
    assert abs((y + bh / 2) - 125) < 1


def test_put_that_on_that_stacks_it(arm_client, to_table):
    f = Finger(to_table).move((150, 250)).hold(0.7)
    t1 = f.t - 0.4
    f.move((325, 125)).hold(0.7)
    t2 = f.t - 0.4
    r = ask(arm_client, samples=f.samples,
            words=words(("put", t1 - 0.25), ("that", t1), ("on", t2 - 0.25), ("that", t2)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"relation": "on",
                               "target": {"deixis": "that", "word_index": 3}})])  # fmt: skip
    place = r.calls[0].where("place")
    assert place.z_mm == 50 and math.dist(place.mean_mm, (325, 125)) < 1  # on the green block


def test_a_bit_left_is_one_arm_step(arm_client):
    prev = {"version": "1.0", "frame": "table", "kind": "box", "mean_mm": [480.0, 90.0],
            "box_mm": [455.0, 65.0, 50.0, 50.0], "source": "gesture"}  # fmt: skip
    r = ask(arm_client, words=words(("a", 8.0), ("bit", 8.25), ("left", 8.5)),
            calls=[call("adjust", dx=-1)], context={"previous": {"place": prev}})  # fmt: skip
    assert r.calls[0].where("place").mean_mm == (450.0, 90.0)


def test_out_of_reach_asks_and_never_moves_it(arm_client, to_table):
    f = Finger(to_table).move((150, 250)).hold(0.7)
    t_that = f.t - 0.4
    f.move((620, 380)).hold(0.7)  # past the table's right edge (the camera still sees it)
    t_here = f.t - 0.4
    r = ask(arm_client, samples=f.samples,
            words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
            calls=[call("move", object={"deixis": "that", "word_index": 1},
                        place={"deixis": "here", "word_index": 2})])  # fmt: skip
    assert r.status == "ask" and r.ask.reason == "outside_area"
    assert r.ask.question == "I can't reach there. Please point somewhere closer."
    assert math.dist(r.ask.candidates[0].mean_mm, (620, 380)) < 8  # as pointed, not moved


def test_between_two_blocks_asks_which_then_the_answer_resolves(arm_client, to_table):
    f = Finger(to_table).move((195, 250)).hold(0.7)  # between red and yellow
    t_that = f.t - 0.4
    f.move((480, 90)).hold(0.7)
    t_here = f.t - 0.4
    req = dict(samples=f.samples,
               words=words(("put", t_that - 0.25), ("that", t_that), ("here", t_here)),
               calls=[call("move", object={"deixis": "that", "word_index": 1},
                           place={"deixis": "here", "word_index": 2})])  # fmt: skip
    first = ask(arm_client, **req)
    assert first.status == "ask" and first.ask.reason == "ambiguous_object"
    assert {c.thing for c in first.ask.candidates} == {"red", "yellow"}
    assert first.ask.question == "This one or that one?"
    pick = [c.thing for c in first.ask.candidates].index("yellow")  # the person clicks it
    back = {"call": 0, "arg": "object", "candidate": pick, "thing": "yellow"}
    second = ask(arm_client, **req,
                 answers=[back, *[a.model_dump(mode="json") for a in first.partial]])  # fmt: skip
    assert second.status == "ok" and second.calls[0].where("object").thing == "yellow"
