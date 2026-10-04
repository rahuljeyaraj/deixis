"""References: how an agent says "here", "that one" or "this big" without coordinates.

Invariant (plan 1.3): the language model never writes coordinates. Every place, size and
object in a tool call is a *reference* (a spoken word's index, a relation to a thing, a
named region, a preset), which `deixis.resolve` turns into a `Where` or a `Size`.

Reference shapes are told apart by their keys, so a model writes compact JSON:

- PlaceRef:  {"deixis": "here", "word_index": 3} | {"pointed": true} |
  {"relation": "left_of", "target": ObjectRef, "target2": ObjectRef?} |
  {"named": "top left corner"} | {"empty_space": true} | {"from": PlaceRef, "to": PlaceRef}
- SizeRef:   {"word_index": 5} | {"preset": "small"} | {"same_as": ObjectRef}
- ObjectRef: {"deixis": "that", "word_index": 7, "label": "cat"?, "count": 2?} |
  {"label": "blue one"} | {"last": true}

A few common slips (a bare "empty_space" string, a bare preset word, "large", "next to") are
repaired before validation, so the core sees one clean shape. Size presets and their
synonyms come from the profile, through the validation context (`ref_context`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationInfo,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from deixis.profile import Profile

#: Words that can carry a pointing gesture ("so" as in "so big"). English only (L7).
DEMONSTRATIVES = ("here", "there", "this", "that", "these", "those", "so")

Demonstrative = Literal["here", "there", "this", "that", "these", "those", "so"]
Relation = Literal[
    "next_to",
    "left_of",
    "right_of",
    "above",
    "below",
    "behind",
    "in_front_of",
    "between",
    "inside",
    "on",
]

RELATION_SYNONYMS = {
    "next to": "next_to",
    "beside": "next_to",
    "near": "next_to",
    "left of": "left_of",
    "right of": "right_of",
    "over": "above",
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "in": "inside",
    "on_top_of": "on",
    "on top of": "on",
    "in front of": "in_front_of",
    "front": "in_front_of",
    "in_front": "in_front_of",
    "back_of": "behind",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def ref_context(profile: Profile) -> dict[str, Any]:
    """The validation context that checks references against a profile's size words."""
    return {
        "size_presets": dict(profile.sizes.presets),
        "size_synonyms": dict(profile.sizes.synonyms),
    }


def _ctx(info: ValidationInfo | None) -> Mapping[str, Any]:
    ctx = info.context if info is not None else None
    return ctx if isinstance(ctx, Mapping) else {}


# ---------------------------------------------------------------- object references


class DeicticObject(_Strict):
    """The thing nearest where the person pointed while saying the word."""

    deixis: Demonstrative = Field(description="The demonstrative word as spoken.")
    word_index: int = Field(ge=0, description="Index of that word in the transcript.")
    label: str | None = Field(
        default=None, description='Only if a name was said with it: "that cat" -> "cat".'
    )
    count: int | None = Field(
        default=None, ge=1, le=2, description='Only for "these two": how many things.'
    )


class LabelObject(_Strict):
    """A thing by its label or attributes, e.g. "house" or "the blue one"."""

    label: str = Field(min_length=1, description="Label or attribute of a thing in the scene.")


class LastObject(_Strict):
    """The most recent thing ("it", "the one you just made")."""

    model_config = ConfigDict(json_schema_extra={"required": ["last"]})

    last: Literal[True] = True


def _object_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {DeicticObject: "deictic", LabelObject: "label", LastObject: "last"}.get(type(v))
    if isinstance(v, dict):
        for key, tag in (("deixis", "deictic"), ("label", "label"), ("last", "last")):
            if key in v:
                return tag
    return None


def _repair_object(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("last", "it", "that one", "the last one"):
            return {"last": True}
        return {"label": v.strip()}
    if isinstance(v, dict) and "word_index" in v and "deixis" not in v:
        return {"deixis": "that", **v}
    if isinstance(v, dict) and isinstance(v.get("deixis"), str):
        d = v["deixis"].strip().lower()
        if d in ("it", "last"):
            return {"last": True}
        if d not in DEMONSTRATIVES:
            return {"label": v["deixis"].strip()}  # {"deixis": "sun"} means the label "sun"
    if isinstance(v, dict) and v.get("deixis") and v.get("label", "") is None:
        v = {k: x for k, x in v.items() if k != "label"}
    return v


ObjectRef = Annotated[
    Annotated[DeicticObject, Tag("deictic")]
    | Annotated[LabelObject, Tag("label")]
    | Annotated[LastObject, Tag("last")],
    Discriminator(
        _object_tag,
        custom_error_type="object_ref",
        custom_error_message='ObjectRef must have one of the keys "deixis", "label" or "last"',
    ),
    BeforeValidator(_repair_object),
]

# ---------------------------------------------------------------- place references


class DeicticPlace(_Strict):
    """Where the person pointed while saying a demonstrative ("here", "there")."""

    deixis: Demonstrative = Field(description="The demonstrative word as spoken.")
    word_index: int = Field(ge=0, description="Index of that word in the transcript.")


class PointedPlace(_Strict):
    """Where the person pointed, with no pointing word ("a tree" while pointing)."""

    model_config = ConfigDict(json_schema_extra={"required": ["pointed"]})

    pointed: Literal[True] = True


class RelationPlace(_Strict):
    """A place relative to things in the scene, computed by geometry in the robot's frame."""

    relation: Relation
    target: ObjectRef
    target2: ObjectRef | None = Field(default=None, description="Only for 'between'.")

    @model_validator(mode="after")
    def _between_needs_two(self) -> RelationPlace:
        pair = isinstance(self.target, DeicticObject) and self.target.count == 2
        if self.relation == "between" and self.target2 is None and not pair:
            raise ValueError("relation 'between' needs target2")
        if self.relation != "between" and self.target2 is not None:
            raise ValueError("target2 is only used with relation 'between'")
        return self


class NamedPlace(_Strict):
    """A place named in words: a region of the work area ("top left corner", "middle"), or a
    part of a thing that needs vision ("on the roof")."""

    named: str = Field(min_length=1)

    @model_validator(mode="after")
    def _not_a_bare_relation(self) -> NamedPlace:
        # Live Nemotron sent {"named": "inside"}, then {"named": "inside that"}, for "put a
        # hand inside that" (1 Oct 2026). A relation word, alone or with a pointing word,
        # names no place: send it back for the retry.
        words = self.named.strip().lower().replace("_", " ").split()
        pointing = ""
        if len(words) > 1 and words[-1] in (*DEMONSTRATIVES, "it", "one"):
            pointing = words.pop()
            if words[-1:] == ["that"] or words[-1:] == ["this"]:  # "inside that one"
                pointing = words.pop()
        word = " ".join(words)
        rel = RELATION_SYNONYMS.get(word, word.replace(" ", "_"))
        if rel in get_args(Relation):
            target = (
                f'{{"deixis": "{pointing}", "word_index": <its index>}}'
                if pointing in DEMONSTRATIVES
                else "<the object it refers to>"
            )
            raise ValueError(
                f"'{self.named}' is a relation, not a place name: use "
                f'{{"relation": "{rel}", "target": {target}}}'
            )
        return self


class EmptySpace(_Strict):
    """The largest empty area."""

    model_config = ConfigDict(json_schema_extra={"required": ["empty_space"]})

    empty_space: Literal[True] = True


def _place_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {
            DeicticPlace: "deictic",
            PointedPlace: "pointed",
            RelationPlace: "relation",
            NamedPlace: "named",
            EmptySpace: "empty",
            TwoPlaces: "two",
        }.get(type(v))
    if isinstance(v, dict):
        for key, tag in (
            ("deixis", "deictic"),
            ("pointed", "pointed"),
            ("relation", "relation"),
            ("named", "named"),
            ("empty_space", "empty"),
            ("from", "two"),
            ("from_", "two"),
        ):
            if key in v:
                return tag
    return None


def _repair_place(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "_")
        if s in ("empty_space", "empty", "anywhere", "somewhere"):
            return {"empty_space": True}
        return {"named": v.strip()}
    if isinstance(v, dict) and "relation" in v and isinstance(v["relation"], str):
        rel = v["relation"].strip().lower()
        v = {**v, "relation": RELATION_SYNONYMS.get(rel, rel.replace(" ", "_"))}
    if isinstance(v, dict) and v.get("target2", "") is None:
        v = {k: x for k, x in v.items() if k != "target2"}
    return v


CornerRef = Annotated[
    Annotated[DeicticPlace, Tag("deictic")]
    | Annotated[PointedPlace, Tag("pointed")]
    | Annotated[RelationPlace, Tag("relation")]
    | Annotated[NamedPlace, Tag("named")],
    Discriminator(
        _place_tag,
        custom_error_type="corner_ref",
        custom_error_message=(
            'Each end of "from ... to" must have one of the keys "deixis", "pointed", '
            '"relation" or "named"'
        ),
    ),
    BeforeValidator(_repair_place),
]


class TwoPlaces(_Strict):
    """Two places: "from here ... to here". Opposite corners of a box, or a path's ends."""

    from_: CornerRef = Field(alias="from")
    to: CornerRef


PlaceRef = Annotated[
    Annotated[DeicticPlace, Tag("deictic")]
    | Annotated[PointedPlace, Tag("pointed")]
    | Annotated[RelationPlace, Tag("relation")]
    | Annotated[NamedPlace, Tag("named")]
    | Annotated[EmptySpace, Tag("empty")]
    | Annotated[TwoPlaces, Tag("two")],
    Discriminator(
        _place_tag,
        custom_error_type="place_ref",
        custom_error_message=(
            'PlaceRef must have one of the keys "deixis", "pointed", "relation", "named", '
            '"empty_space" or "from"'
        ),
    ),
    BeforeValidator(_repair_place),
]

# ---------------------------------------------------------------- size references


class DeicticSize(_Strict):
    """A size shown with the hands while saying "this big" (index of "this")."""

    word_index: int = Field(ge=0, description='Index of "this"/"that"/"so" in "this big".')
    deixis: Demonstrative | None = None


def norm_preset(s: str, synonyms: Mapping[str, str] | None = None) -> str:
    """'Large ' -> 'big' (with the profile's synonyms); spaces become underscores."""
    s = s.strip().lower()
    syn = synonyms or {}
    return syn.get(s, syn.get(s.replace(" ", "_"), s.replace(" ", "_")))


class PresetSize(_Strict):
    """A spoken size word; the profile lists the words (`sizes.presets`)."""

    preset: str = Field(min_length=1)

    @field_validator("preset", mode="before")
    @classmethod
    def _known(cls, v: Any, info: ValidationInfo) -> Any:
        if not isinstance(v, str):
            return v
        ctx = _ctx(info)
        v = norm_preset(v, ctx.get("size_synonyms"))
        presets = ctx.get("size_presets")
        if presets is not None and v not in presets:
            names = ", ".join(f"'{p}'" for p in presets) or "none"
            raise ValueError(f"unknown size preset '{v}'; presets: {names}")
        return v


class RelativeSize(_Strict):
    """The same size as another thing."""

    same_as: ObjectRef


def _size_tag(v: Any) -> str | None:
    if isinstance(v, BaseModel):
        return {DeicticSize: "deictic", PresetSize: "preset", RelativeSize: "relative"}.get(type(v))
    if isinstance(v, dict):
        for key, tag in (("preset", "preset"), ("same_as", "relative"), ("word_index", "deictic")):
            if key in v:
                return tag
    return None


def _repair_size(v: Any) -> Any:
    if isinstance(v, str):
        return {"preset": v}
    return v


SizeRef = Annotated[
    Annotated[DeicticSize, Tag("deictic")]
    | Annotated[PresetSize, Tag("preset")]
    | Annotated[RelativeSize, Tag("relative")],
    Discriminator(
        _size_tag,
        custom_error_type="size_ref",
        custom_error_message='SizeRef must have one of the keys "word_index", "preset", "same_as"',
    ),
    BeforeValidator(_repair_size),
]

# ---------------------------------------------------------------- walking references

_PUNCT = re.compile(r"[^\w']+")


def norm_word(w: str) -> str:
    """Lowercase and strip punctuation: 'Here.' -> 'here'."""
    return _PUNCT.sub("", w.lower())


def word_refs(obj: Any, path: str = "") -> list[tuple[str, int, str | None]]:
    """All (path, word_index, deixis) pairs of deictic references inside a model."""
    out: list[tuple[str, int, str | None]] = []
    if isinstance(obj, (DeicticPlace, DeicticObject, DeicticSize)):
        out.append((path, obj.word_index, obj.deixis))
    elif isinstance(obj, BaseModel):
        for field in type(obj).model_fields:
            out += word_refs(getattr(obj, field), f"{path}.{field}" if path else field)
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            out += word_refs(x, f"{path}[{i}]")
    return out


def label_refs(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """All (path, label) pairs of label object references inside a model."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, LabelObject):
        out.append((path, obj.label))
    elif isinstance(obj, BaseModel):
        for field in type(obj).model_fields:
            out += label_refs(getattr(obj, field), f"{path}.{field}" if path else field)
    elif isinstance(obj, (list, tuple)):
        for i, x in enumerate(obj):
            out += label_refs(x, f"{path}[{i}]")
    return out


#: Words that are never a thing's label (a model once sent {"label": "the"}, 30 Sep 2026).
NOT_NAMES = {"the", "a", "an", "this", "that", "these", "those", "it", "one", "them", "to", ""}


def transcript_problems(name: str, args: BaseModel, words: Sequence[str]) -> list[str]:
    """Problems with a call's references given the transcript: word indices that don't point
    at a demonstrative, and labels that are not names. Empty list means fine."""
    problems = []
    normed = [norm_word(w) for w in words]
    for path, idx, deixis in word_refs(args):
        if idx >= len(words):
            problems.append(f"{name}.{path}: word_index {idx} is past the last word")
        elif normed[idx] not in DEMONSTRATIVES:
            hint = ""
            if deixis and deixis in normed:
                hint = f"; '{deixis}' is word {normed.index(deixis)}"
            problems.append(
                f"{name}.{path}: word {idx} is '{words[idx]}', not a demonstrative{hint}"
            )
        elif deixis and normed[idx] != deixis:
            problems.append(
                f"{name}.{path}: word {idx} is '{words[idx]}' but deixis says '{deixis}'"
            )
    for path, label in label_refs(args):
        if norm_word(label) in NOT_NAMES:
            problems.append(
                f"{name}.{path}: '{label}' is not a name; use the thing's label "
                'from the Scene line, e.g. {"label": "house"}'
            )
    return problems


def with_presets(schema: Any, presets: Sequence[str]) -> Any:
    """A tool's JSON Schema with the size preset words listed (for the agent)."""
    if isinstance(schema, dict):
        out = {k: with_presets(v, presets) for k, v in schema.items()}
        props = out.get("properties")
        if isinstance(props, dict) and isinstance(props.get("preset"), dict):
            props["preset"] = {**props["preset"], "enum": list(presets)}
        return out
    if isinstance(schema, list):
        return [with_presets(x, presets) for x in schema]
    return schema
