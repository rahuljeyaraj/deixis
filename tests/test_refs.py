"""Reference shapes, repairs and transcript checks (C6)."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from deixis.refs import (
    DeicticObject,
    DeicticPlace,
    DeicticSize,
    EmptySpace,
    LabelObject,
    LastObject,
    NamedPlace,
    ObjectRef,
    PlaceRef,
    PointedPlace,
    PresetSize,
    RelationPlace,
    RelativeSize,
    SizeRef,
    TwoPlaces,
    norm_word,
    ref_context,
    transcript_problems,
    with_presets,
)


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place: PlaceRef | None = None
    object: ObjectRef | None = None
    size: SizeRef | None = None


def args(context=None, **kw):
    return Args.model_validate(kw, context=context)


@pytest.mark.parametrize(
    "place,cls",
    [
        ({"deixis": "here", "word_index": 3}, DeicticPlace),
        ({"pointed": True}, PointedPlace),
        ({"relation": "left_of", "target": {"label": "house"}}, RelationPlace),
        ({"named": "top left corner"}, NamedPlace),
        ({"empty_space": True}, EmptySpace),
        ("empty_space", EmptySpace),
        ("the middle", NamedPlace),
        (
            {
                "from": {"deixis": "here", "word_index": 2},
                "to": {"deixis": "here", "word_index": 5},
            },
            TwoPlaces,
        ),
    ],
)
def test_place_shapes(place, cls):
    assert isinstance(args(place=place).place, cls)


def test_two_places_keep_their_order_and_dump_with_from():
    a = args(place={"from": {"deixis": "here", "word_index": 2}, "to": {"pointed": True}})
    assert a.place.from_ == DeicticPlace(deixis="here", word_index=2)
    assert isinstance(a.place.to, PointedPlace)
    assert a.model_dump(by_alias=True, exclude_none=True)["place"] == {
        "from": {"deixis": "here", "word_index": 2},
        "to": {"pointed": True},
    }
    with pytest.raises(ValidationError, match="from ... to"):
        args(place={"from": {"empty_space": True}, "to": {"pointed": True}})


def test_object_shapes_and_repairs():
    assert args(object={"deixis": "that", "word_index": 1}).object == DeicticObject(
        deixis="that", word_index=1
    )
    that_cat = args(object={"deixis": "that", "word_index": 1, "label": "cat"}).object
    assert that_cat.label == "cat" and that_cat.count is None
    assert isinstance(args(object="it").object, LastObject)
    assert args(object="blue one").object == LabelObject(label="blue one")
    assert args(object={"deixis": "sun", "word_index": 3}).object == LabelObject(label="sun")
    assert args(object={"word_index": 4}).object.deixis == "that"
    with pytest.raises(ValidationError):
        args(object={"deixis": "these", "word_index": 1, "count": 3})


def test_between_takes_two_targets_or_these_two():
    with pytest.raises(ValidationError, match="target2"):
        args(place={"relation": "between", "target": {"label": "a"}})
    two = {"deixis": "these", "word_index": 4, "count": 2}
    p = args(place={"relation": "between", "target": two}).place
    assert p.target.count == 2 and p.target2 is None
    with pytest.raises(ValidationError, match="only used"):
        args(place={"relation": "on", "target": {"label": "a"}, "target2": {"label": "b"}})


@pytest.mark.parametrize(
    "said,rel",
    [
        ("next to", "next_to"),
        ("in front of", "in_front_of"),
        ("under", "below"),
        ("behind", "behind"),
    ],
)
def test_relation_synonyms(said, rel):
    assert args(place={"relation": said, "target": "house"}).place.relation == rel


def test_a_bare_relation_is_not_a_named_place():
    for word in ("inside", "next to", "in front of", "inside that", "next to it"):
        with pytest.raises(ValidationError, match="relation, not a place name"):
            args(place={"named": word})
    for ok in ("top left corner", "middle", "on the roof"):
        assert isinstance(args(place={"named": ok}).place, NamedPlace)


def test_presets_come_from_the_profile(plotter, vacuum, arm):
    ctx = ref_context(plotter)
    assert args(ctx, size="large").size == PresetSize(preset="big")
    assert args(ctx, size={"preset": "Whole page"}).size == PresetSize(preset="whole_page")
    assert args(ref_context(vacuum), size="large").size == PresetSize(preset="big")
    with pytest.raises(ValidationError, match="unknown size preset 'tiny'"):
        args(ref_context(vacuum), size="tiny")
    with pytest.raises(ValidationError, match="presets: none"):
        args(ref_context(arm), size="small")
    assert args(size="Very Big").size == PresetSize(preset="very_big")  # no profile: as said


def test_size_shapes():
    assert isinstance(args(size={"word_index": 5}).size, DeicticSize)
    assert isinstance(
        args(size={"same_as": {"deixis": "that", "word_index": 9}}).size, RelativeSize
    )


def test_transcript_problems():
    words = "put a tree here this big".split()
    ok = args(place={"deixis": "here", "word_index": 3}, size={"word_index": 4})
    assert transcript_problems("draw", ok, words) == []
    (msg,) = transcript_problems("draw", args(place={"deixis": "here", "word_index": 2}), words)
    assert "word 2 is 'tree'" in msg and "'here' is word 3" in msg
    far = args(place={"deixis": "here", "word_index": 9})
    assert "past the last word" in transcript_problems("draw", far, words)[0]
    the = args(object={"label": "the"})
    assert "not a name" in transcript_problems("move", the, words)[0]
    corners = args(
        place={
            "from": {"deixis": "here", "word_index": 3},
            "to": {"deixis": "here", "word_index": 1},
        }
    )
    (msg,) = transcript_problems("clean", corners, words)
    assert "place.to: word 1 is 'a'" in msg


def test_with_presets_lists_the_words():
    schema = Args.model_json_schema()
    text = str(with_presets(schema, ["small", "big"]))
    assert "'enum': ['small', 'big']" in text


def test_norm_word():
    assert norm_word("Here.") == "here" and norm_word("That's") == "that's"
