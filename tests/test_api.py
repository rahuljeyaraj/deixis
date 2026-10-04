"""resolve() on the plan's worked examples E1 to E7 (plan 4.11, C11)."""

import json

import pytest

from deixis.api import Request, Response, adjust, adjust_size, resolve
from deixis.where import Answer, Size, Where
from support import ROOT, profile

EXAMPLES = ROOT / "examples" / "requests"


def run(name, **changes):
    d = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    d["request"].update(changes)
    return resolve(d["request"], profile(d["profile"])), d


def test_e1_plotter_cat_here_this_big():
    r, _ = run("E1-plotter-cat-here-this-big")
    assert r.status == "ok" and r.ask is None
    (call,) = r.calls
    assert call.name == "place_drawing" and call.args["object"] == "cat"
    place, size = call.where("place"), call.size("size")
    assert place.frame == "paper" and place.kind == "point" and place.mean_mm == (112.4, 80.2)
    assert place.cov_mm2 == ((2.89, 0.0), (0.0, 3.06)) and place.source == "gesture"
    assert place.confidence == 1.0
    ev = place.evidence
    assert ev["rule"] == "nearest" and ev["word"] == "here" and ev["word_t"] == [10.8, 11.1]
    assert ev["hold_t"] == [10.2, 10.95] and ev["lead_s"] == 0.225 and ev["candidates"] == 1
    assert size.mm == 52.0 and size.side == "larger" and size.source == "gesture"
    assert size.evidence["word"] == "this" and size.evidence["size_kind"] == "pinch"
    assert [set(e) for e in r.log] == [{"size", "size_kind", "word"}, {"where"}]  # size first


def test_e2_vacuum_clean_here():
    r, _ = run("E2-vacuum-clean-here")
    place = r.calls[0].where("place")
    assert place.frame == "floor" and place.kind == "box"
    assert place.box_mm == (1020, 480, 1000, 1000) and place.mean_mm == (1520, 980)
    assert place.source == "gesture" and place.evidence["rule"] == "nearest"
    assert place.evidence["word"] == "here" and place.evidence["box_from"] == "default"
    assert place.evidence["size_mm"] == 1000 and len(r.holds) == 1


def test_e3_vacuum_clean_from_here_to_here():
    r, _ = run("E3-vacuum-clean-from-here-to-here")
    place = r.calls[0].where("place")
    assert place.kind == "box" and place.box_mm == (800, 600, 1300, 900)
    assert place.evidence["rule"] == "k_in_order"
    assert place.evidence["corners"] == [[800, 600], [2100, 1500]]


def test_e4_arm_put_that_here():
    r, _ = run("E4-arm-put-that-here")
    call = r.calls[0]
    obj, place = call.where("object"), call.where("place")
    assert obj.kind == "box" and obj.box_mm == (125, 225, 50, 50)
    assert obj.thing == "red" and obj.label == "red block" and obj.source == "object_ref"
    assert place.kind == "box" and place.box_mm == (455, 65, 50, 50)
    assert place.mean_mm == (480, 90) and place.source == "gesture"
    assert place.evidence["box_from"] == "object"


def test_e5_arm_unsure_then_the_answer():
    r, d = run("E5-arm-unsure")
    assert r.status == "ask" and r.calls == []
    ask = r.ask
    assert ask.reason == "ambiguous_object" and ask.question == "This one or that one?"
    assert (ask.call, ask.arg) == (0, "object")
    assert [c.thing for c in ask.candidates] == ["red", "yellow"]
    assert ask.options == ["red block", "yellow block"] and ask.word.text == "that"
    again, _ = run("E5-arm-unsure", answers=[{"call": 0, "arg": "object", "thing": "red"}])
    assert again.status == "ok" and again.calls[0].where("object").thing == "red"
    assert again.calls[0].where("place").box_mm == (455, 65, 50, 50)
    chosen = ask.candidates[1]
    by_candidate, _ = run(
        "E5-arm-unsure", answers=[{"call": 0, "arg": "object", "candidate": 1, "where": chosen}]
    )
    assert by_candidate.calls[0].where("object").thing == "yellow"


def test_e6_arm_a_bit_left():
    r, _ = run("E6-arm-a-bit-left")
    call = r.calls[0]
    assert call.name == "adjust" and call.args["dx"] == -1
    place = call.where("place")
    assert place.box_mm == (425, 65, 50, 50) and place.mean_mm == (450, 90)
    assert place.source == "adjust" and place.evidence["previous"]["box_mm"] == [455, 65, 50, 50]


def test_e7_arm_put_that_on_that():
    r, _ = run("E7-arm-put-that-on-that")
    place = r.calls[0].where("place")
    assert place.kind == "box" and place.box_mm == (300, 100, 50, 50) and place.z_mm == 50
    assert place.evidence["relation"] == "on" and place.evidence["target"] == "green"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("E*.json")), ids=lambda p: p.stem[:2])
def test_same_input_gives_byte_identical_output(path):
    d = json.loads(path.read_text(encoding="utf-8"))
    p = profile(d["profile"])
    a = resolve(d["request"], p).model_dump_json()
    b = resolve(Request.model_validate_json(json.dumps(d["request"])), p).model_dump_json()
    assert a == b
    assert Response.model_validate_json(a).model_dump_json() == a  # round trip


def test_adjust_functions(arm):
    w = Where(frame="table", kind="point", mean_mm=(100, 100), source="gesture")
    assert adjust(w, dx=1, dy=-2, profile=arm).mean_mm == (130, 40)
    box = Where(frame="table", kind="box", box_mm=(80, 80, 40, 40), source="gesture")
    big = adjust(box, scale=1.5, profile=arm)
    assert big.box_mm == (70, 70, 60, 60) and big.mean_mm == (100, 100)
    far = adjust(w, dx=20, profile=arm)
    assert far.mean_mm == (700, 100) and far.evidence["outside"] is True  # flagged, not moved
    s = adjust_size(Size(mm=40, source="gesture"), 1.25)
    assert s.mm == 50 and s.source == "adjust" and s.evidence["previous"]["mm"] == 40
    with pytest.raises(ValueError, match="profile"):
        adjust(w, dx=1)


def test_partial_comes_back_on_ask(arm):
    """The object is found, the place is not: the object is in `partial`, to send back."""
    r, d = run("E4-arm-put-that-here", holds=[d_ for d_ in json.loads(
        (EXAMPLES / "E4-arm-put-that-here.json").read_text())["request"]["holds"][:1]])  # fmt: skip
    assert r.status == "ask" and r.ask.reason == "no_gesture" and r.ask.arg == "place"
    (got,) = r.partial
    assert got.arg == "object" and got.where.thing == "red"
    hold = {"t0": 3.2, "t1": 3.9, "mean": [480, 90], "spread": [1, 1], "frame": "table"}
    answers = [got.model_dump(mode="json"), Answer(call=0, arg="place", hold=hold)]
    again, _ = run("E4-arm-put-that-here", holds=[], answers=[
        a if isinstance(a, dict) else a.model_dump(mode="json") for a in answers])  # fmt: skip
    assert again.status == "ok"
    assert again.calls[0].where("place").box_mm == (455, 65, 50, 50)
