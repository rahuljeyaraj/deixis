"""Actions, tools for any agent, parsing and checking the agent's calls (C10)."""

import json

import jsonschema
import pytest

from deixis.actions import (
    CallError,
    check_call,
    field_name,
    parse_call,
    tools_for_agent,
)
from deixis.refs import EmptySpace, LabelObject, LastObject, PresetSize, TwoPlaces
from support import PROFILE_NAMES, profile

COORD_KEYS = {"x", "y", "x_mm", "y_mm", "mm", "coords", "coordinates", "point", "box", "mean_mm"}

EXAMPLE_CALLS = {
    "plotter-a5": [
        ("place_drawing", {"object": "cat", "place": {"deixis": "here", "word_index": 3},
                           "size": {"word_index": 4}}),
        ("draw_connector", {"kind": "arrow", "from": {"deixis": "this", "word_index": 2},
                            "to": {"label": "fish"}}),
        ("write_label", {"text": "Cosmos",
                         "place": {"relation": "below", "target": {"last": True}}}),
    ],
    "vacuum-floor": [
        ("clean", {"place": {"deixis": "here", "word_index": 1}}),
        ("clean", {"place": {"from": {"deixis": "here", "word_index": 2},
                             "to": {"deixis": "here", "word_index": 5}}}),
        ("dock", {}),
    ],
    "arm-table": [
        ("move", {"object": {"deixis": "that", "word_index": 1},
                  "place": {"deixis": "here", "word_index": 2}}),
        ("move", {"object": {"label": "blue one"},
                  "place": {"relation": "next_to", "target": {"deixis": "that", "word_index": 6}}}),
        ("move", {"object": {"last": True}, "place": {"relation": "on",
                  "target": {"deixis": "that", "word_index": 3}}}),
        ("adjust", {"dx": -1}),
    ],
}  # fmt: skip


def _keys(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "properties":
                out.update(v.keys())
            _keys(v, out)
    elif isinstance(node, list):
        for x in node:
            _keys(x, out)
    return out


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_tools_are_valid_json_schema_without_coordinates(name):
    p = profile(name)
    tools = tools_for_agent(p)
    names = [t["function"]["name"] for t in tools]
    assert names[: len(p.actions)] == [a.name for a in p.actions]
    assert {"adjust", "confirm", "cancel", "stop", "ask_user"} <= set(names)
    assert "locate" not in names  # resolvable, not offered to agents
    for t in tools:
        params = t["function"]["parameters"]
        jsonschema.Draft202012Validator.check_schema(params)
        assert params["type"] == "object" and t["function"]["description"]
        text = json.dumps(params)
        assert "$ref" not in text and "$defs" not in text and '"title"' not in text
    assert not _keys(tools, set()) & COORD_KEYS


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_example_calls_validate_against_the_tools_and_parse(name):
    p = profile(name)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools_for_agent(p)}
    for tool, args in EXAMPLE_CALLS[name]:
        jsonschema.validate(args, schemas[tool])
        call = parse_call(p, tool, json.dumps(args))
        assert call.name == tool and call.to_dict()["args"].items() >= args.items()


def test_size_shapes_follow_the_profile(plotter, vacuum):
    sizes = {
        t["function"]["name"]: json.dumps(t["function"]["parameters"]["properties"].get("size"))
        for t in tools_for_agent(plotter)
    }
    assert '"enum": ["tiny", "small", "medium", "big", "whole_page"]' in sizes["place_drawing"]
    assert "this big" in sizes["place_drawing"]
    d = vacuum.model_dump()
    d["actions"].append({"name": "spot", "params": [{"name": "size", "type": "size"}]})
    no_gesture = type(vacuum).model_validate(d)
    spot = next(t for t in tools_for_agent(no_gesture) if t["function"]["name"] == "spot")
    size = json.dumps(spot["function"]["parameters"]["properties"]["size"])
    assert "this big" not in size and '"enum": ["small", "medium", "big"]' in size


def test_the_plotters_repairs_still_work(plotter):
    call = parse_call(plotter, "place_drawing", {"object": "sun", "place": "empty_space",
                                                 "size": "large"})  # fmt: skip
    assert isinstance(call.ref("place"), EmptySpace) and call.ref("size") == PresetSize(
        preset="big"
    )
    call = parse_call(plotter, "draw_connector", {"from": {"deixis": "sun", "word_index": 3},
                                                  "to": {"deixis": "it"}})  # fmt: skip
    assert call.ref("from") == LabelObject(label="sun") and isinstance(call.ref("to"), LastObject)
    call = parse_call(plotter, "place_drawing", {"object": "cat", "place": {
        "relation": "next to", "target": "house", "target2": None}})  # fmt: skip
    assert call.ref("place").relation == "next_to"
    call = parse_call(plotter, "ask_user", {"question": "Which?", "options": list("abcdef")})
    assert len(call.ref("options")) == 4
    for name in ("confirm", "cancel", "stop", "new_sheet"):
        assert parse_call(plotter, name, "").args_dict() == {}


@pytest.mark.parametrize(
    "name,args,match",
    [
        ("fly", {}, "unknown tool 'fly'"),
        ("clean", "{not json", "not valid JSON"),
        ("clean", "[1]", "JSON object"),
        ("clean", {}, "place: Field required"),
        ("clean", {"place": {"foo": 1}}, "PlaceRef"),
        ("clean", {"place": {"pointed": True}, "x": 3}, "Extra inputs"),
        ("adjust", {"dx": 20}, "less than or equal"),
        ("dock", {"now": True}, "Extra inputs"),
        ("ask_user", {"question": ""}, "at least 1"),
    ],
)
def test_bad_calls_give_readable_errors(vacuum, name, args, match):
    with pytest.raises(CallError, match=match):
        parse_call(vacuum, name, args)


def test_unknown_tool_lists_the_agents_tools(vacuum):
    with pytest.raises(CallError) as e:
        parse_call(vacuum, "draw", {})
    assert "clean, dock, adjust" in str(e.value) and "locate" not in str(e.value)
    assert parse_call(vacuum, "locate", {"place": {"deixis": "here", "word_index": 0}})


def test_unknown_preset_for_this_robot(vacuum, plotter):
    vac = vacuum.model_copy(deep=True)
    vac.actions[0].params.append(type(vac.actions[0].params[0])(name="size", type="size"))
    with pytest.raises(CallError, match="unknown size preset 'whole_page'"):
        parse_call(vac, "clean", {"place": {"pointed": True}, "size": "whole page"})
    assert parse_call(plotter, "place_drawing", {"object": "x", "size": "the whole page"})


def test_check_call_against_the_words(vacuum):
    words = "clean from here to here".split()
    corners = {"from": {"deixis": "here", "word_index": 2},
               "to": {"deixis": "here", "word_index": 4}}  # fmt: skip
    ok = parse_call(vacuum, "clean", {"place": corners})
    assert isinstance(ok.ref("place"), TwoPlaces) and check_call(ok, words) == []
    bad = parse_call(vacuum, "clean", {"place": {"deixis": "here", "word_index": 1}})
    (msg,) = check_call(bad, words)
    assert "word 1 is 'from'" in msg and "'here' is word 2" in msg


def test_field_names():
    assert field_name("from") == "from_" and field_name("place") == "place"
    assert field_name("2d") == "p_2d"
