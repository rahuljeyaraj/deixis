"""Where 1.0, Size, Ask, Answer and the committed JSON Schemas (C5)."""

import json
import tomllib

import pytest
from pydantic import ValidationError

import deixis
from deixis.resolver import Pick, hold_where
from deixis.schema import file_name, schema_text, schemas
from deixis.types import Hold, Word
from deixis.where import Answer, Ask, Size, Where
from support import ROOT

#: A `where` exactly as the plotter logged it before version 1.0 (live log, 2 Oct 2026).
OLD_WHERE = {
    "frame": "paper",
    "kind": "point",
    "mean_mm": [112.4, 80.2],
    "box_mm": None,
    "path_mm": None,
    "cov_mm2": [[2.89, 0.0], [0.0, 3.06]],
    "source": "gesture",
    "confidence": 1.0,
    "evidence": {
        "rule": "nearest",
        "hold_t": [10.2, 10.95],
        "candidates": 1,
        "word": "here",
        "word_t": [10.8, 11.1],
        "lead_s": 0.225,
    },
}


def test_committed_schemas_are_current():
    """Change a wire type on purpose: bump the minor version, `python -m deixis schema`,
    and add a CHANGELOG line."""
    for name, schema in schemas().items():
        path = ROOT / "schema" / file_name(name)
        assert path.exists(), f"missing {path}; run: uv run python -m deixis schema"
        assert path.read_text(encoding="utf-8") == schema_text(schema), (
            f"{path.name} is out of date; run: uv run python -m deixis schema"
        )


def test_old_where_json_still_validates():
    w = Where.model_validate(OLD_WHERE)
    assert w.version == "1.0" and w.mean_mm == (112.4, 80.2)
    assert w.z_mm is None and w.thing is None and w.label is None


def test_where_fills_the_mean_and_round_trips():
    box = Where(frame="floor", kind="box", box_mm=(1020, 480, 1000, 1000), source="gesture")
    assert box.mean_mm == (1520, 980)
    path = Where(frame="paper", kind="path", path_mm=[(0, 0), (10, 0), (20, 30)], source="geometry")
    assert path.mean_mm == (10, 10)
    obj = Where(
        frame="table", kind="box", box_mm=(125, 225, 50, 50), z_mm=50, thing="red",
        label="red block", source="object_ref",
    )  # fmt: skip
    assert Where.model_validate_json(obj.model_dump_json()) == obj


def test_readers_ignore_unknown_fields_and_refuse_version_2():
    w = Where.model_validate({**OLD_WHERE, "version": "1.3", "angle_deg": 30})
    assert w.version == "1.3" and "angle_deg" not in w.model_dump()
    with pytest.raises(ValidationError, match="not 1.x"):
        Where.model_validate({**OLD_WHERE, "version": "2.0"})


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "point"},
        {"kind": "box"},
        {"kind": "box", "box_mm": (0, 0, -1, 5)},
        {"kind": "path", "path_mm": [(0, 0)]},
        {"kind": "point", "mean_mm": (0, 0), "source": "magic"},
    ],
)
def test_where_rejects_incomplete(bad):
    with pytest.raises(ValidationError):
        Where.model_validate({"frame": "f", "source": "gesture", **bad})


def test_size_and_ask():
    s = Size(mm=52.0, source="gesture", evidence={"word": "this", "size_kind": "pinch"})
    assert s.side == "larger" and Size.model_validate_json(s.model_dump_json()) == s
    with pytest.raises(ValidationError):
        Size(mm=0, source="default")
    red = Where(frame="table", kind="box", box_mm=(125, 225, 50, 50), source="object_ref")
    ask = Ask(
        reason="ambiguous_object", question="This one or that one?", call=0, arg="object",
        candidates=[red], options=["red block"], word=Word("that", 1.0, 1.2),
    )  # fmt: skip
    again = Ask.model_validate(json.loads(ask.model_dump_json()))
    assert again == ask and again.word == Word("that", 1.0, 1.2)
    with pytest.raises(ValidationError):
        Ask(reason="confused", question="?")


def test_answer_carries_one_value():
    h = Hold(1.0, 1.5, (10.0, 20.0), (0.5, 0.5), "table", "click")
    a = Answer.model_validate({"call": 0, "arg": "place", "hold": json.loads(json.dumps(
        {"t0": 1.0, "t1": 1.5, "mean": [10, 20], "spread": [0.5, 0.5], "frame": "table",
         "source": "click"}))})  # fmt: skip
    assert a.hold == h
    assert Answer(arg="object", candidate=1).candidate == 1
    assert Answer(arg="object", candidate=0, thing="red").thing == "red"
    with pytest.raises(ValidationError, match="one value"):
        Answer(arg="object", thing="red", size=Size(mm=5, source="default"))
    with pytest.raises(ValidationError, match="needs a candidate"):
        Answer(arg="object")


def test_a_hold_with_its_own_covariance():
    """F14: a gaze tracker passes its own accuracy; it replaces spread plus base sigma."""
    word = Word("there", 2.0, 2.2)
    plain = Hold(1.5, 2.1, (300.0, 200.0), (1.0, 1.0), "table", "gaze")
    gaze = Hold(1.5, 2.1, (300.0, 200.0), (1.0, 1.0), "table", "gaze", ((400.0, 0), (0, 900.0)))
    cov = hold_where(Pick(plain, "nearest", word)).cov_mm2
    assert cov[0][0] == pytest.approx(3.25) and cov[1][1] == pytest.approx(3.25)
    w = hold_where(Pick(gaze, "nearest", word))
    assert w.cov_mm2 == ((400, 0), (0, 900)) and w.frame == "table"


def test_package_version_matches_pyproject():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert deixis.__version__ == meta["project"]["version"] == "0.1.0"
