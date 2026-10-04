"""What a robot can do: its actions and their parameters, declared in its profile.

The core uses them to know each argument's type when resolving (a `place`, an `object`, a
`size`, or something it only passes through), to build the tool list for any agent
(`tools_for_agent`), and to validate the agent's calls (`parse_call`, `check_call`). The
robot decides what each action does.
"""

from __future__ import annotations

import copy
import json
import keyword
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)

from deixis.refs import (
    ObjectRef,
    PlaceRef,
    SizeRef,
    ref_context,
    transcript_problems,
    with_presets,
)

if TYPE_CHECKING:
    from deixis.profile import Profile

ParamType = Literal["place", "object", "size", "text", "choice", "steps", "scale", "flag", "list"]
Want = Literal["point", "box", "path"]
SizeFrom = Literal["size_param", "object", "default"]

#: Parameter types the core resolves; the others are copied through unchanged.
REFERENCE_TYPES = ("place", "object", "size")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Param(_Model):
    """One argument of an action."""

    name: str = Field(min_length=1)
    type: ParamType
    want: Want = Field(
        default="point", description="place only: the shape the robot needs (point, box, path)."
    )
    size_from: SizeFrom | None = Field(
        default=None,
        description=(
            "place with want=box: where the box size comes from. Default: the action's size "
            "parameter if it has one, else the profile's default size."
        ),
    )
    allow_point: bool = Field(
        default=False,
        description=(
            "object only: when nothing is near the pointed spot, give the point itself "
            "instead of asking (an arrow from a thing or from an empty spot)."
        ),
    )
    required: bool = False
    choices: list[str] = Field(default_factory=list, description="choice only.")
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> Param:
        if self.type == "choice" and not self.choices:
            raise ValueError(f"param '{self.name}': a choice needs choices")
        return self


class Action(_Model):
    """Something the robot can do, as the agent sees it."""

    name: str = Field(min_length=1, description="e.g. 'move'; may be 'robot.action' (two robots).")
    description: str = ""
    params: list[Param] = Field(default_factory=list)
    agent: bool = Field(
        default=True, description="false: resolvable, but not offered to agents as a tool."
    )

    @model_validator(mode="after")
    def _unique(self) -> Action:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"action '{self.name}': parameter names must be unique")
        return self

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


#: Actions every profile gets. They need no robot code to resolve; the robot decides what
#: they do. `locate` is not offered to agents: a robot uses it to turn an answer ("here",
#: while pointing again) into a place or a hold.
STANDARD_ACTIONS: tuple[Action, ...] = (
    Action(
        name="adjust",
        description=(
            "Move or resize what is being previewed: dx/dy in steps (1 = a bit, 3 = a lot; "
            "+dx to the person's right, -dx left, +dy away from them, -dy toward them), scale "
            "as a factor (1.25 bigger, 0.8 smaller)."
        ),
        params=[
            Param(name="dx", type="steps", description="Steps to the person's right (+) or left."),
            Param(name="dy", type="steps", description="Steps away from the person (+) or toward."),
            Param(name="scale", type="scale", description="Size factor, 1 = unchanged."),
        ],
    ),
    Action(name="confirm", description="The person accepts the preview (yes, ok, do it)."),
    Action(name="cancel", description="The person rejects the preview or the request (no)."),
    Action(name="stop", description="Stop the robot immediately (stop, halt, wait, freeze)."),
    Action(
        name="ask_user",
        description="Ask one short question, only when the request is really ambiguous.",
        params=[
            Param(name="question", type="text", required=True),
            Param(name="options", type="list", description="2 to 4 short options."),
        ],
    ),
    Action(
        name="locate",
        description="Where the person pointed: resolves one place.",
        params=[Param(name="place", type="place", want="point", required=True)],
        agent=False,
    ),
)

# ---------------------------------------------------------------- calls


class CallError(ValueError):
    """A tool call the model got wrong; the message is written for the model to read."""


class Call(BaseModel):
    """One validated call: the action's name and its argument model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    args: BaseModel

    def args_dict(self) -> dict[str, Any]:
        return self.args.model_dump(by_alias=True, exclude_none=True, mode="json")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args_dict()}

    def ref(self, param: str) -> Any:
        """The validated value of one parameter (a reference model, or a plain value)."""
        return getattr(self.args, field_name(param))

    def __str__(self) -> str:  # compact, for logs
        return f"{self.name}({json.dumps(self.args_dict(), separators=(',', ':'))})"


def field_name(param: str) -> str:
    """The Python field for a parameter name ('from' -> 'from_')."""
    name = param.replace(".", "_").replace("-", "_")
    if keyword.iskeyword(name):
        return f"{name}_"
    if not name.isidentifier():
        return f"p_{name}"
    return name


def _options(v: Any) -> Any:
    return v[:4] if isinstance(v, list) else v


def _field(p: Param) -> tuple[Any, Any]:
    kw: dict[str, Any] = {"alias": p.name}
    if p.description:
        kw["description"] = p.description
    if p.type == "steps":
        return int, Field(default=0, ge=-5, le=5, **kw)
    if p.type == "scale":
        return float, Field(default=1.0, ge=0.25, le=4.0, **kw)
    if p.type == "flag":
        return bool, Field(default=False, **kw)
    if p.type == "list":  # short strings (ask_user's options)
        typ: Any = Annotated[list[str], BeforeValidator(_options)]
        return typ, Field(default_factory=list, max_length=4, **kw)
    typ = {
        "place": PlaceRef,
        "object": ObjectRef,
        "size": SizeRef,
        "text": Annotated[str, Field(min_length=1)],
    }.get(p.type)
    if typ is None:  # choice
        typ = Literal[tuple(p.choices)]  # type: ignore[valid-type]
    if p.required:
        return typ, Field(..., **kw)
    return typ | None, Field(default=None, **kw)


_MODELS: dict[str, type[BaseModel]] = {}


def args_model(action: Action) -> type[BaseModel]:
    """A Pydantic model for an action's arguments (cached by the action's definition)."""
    key = action.model_dump_json()
    model = _MODELS.get(key)
    if model is None:
        fields = {field_name(p.name): _field(p) for p in action.params}
        title = "".join(w.capitalize() for w in action.name.replace(".", "_").split("_"))
        model = create_model(  # type: ignore[call-overload]
            f"{title}Args",
            __config__=ConfigDict(extra="forbid", populate_by_name=True),
            __doc__=action.description or None,
            **fields,
        )
        _MODELS[key] = model
    return model


def parse_call(profile: Profile, name: str, args: dict[str, Any] | str | None) -> Call:
    """Validate one raw tool call against the profile. Raises `CallError` with a message
    written for the model (so the agent loop can send it back for a retry)."""
    action = profile.action(name)
    if action is None:
        names = ", ".join(a.name for a in profile.all_actions() if a.agent)
        raise CallError(f"unknown tool '{name}'. Tools: {names}")
    if args is None or args == "":
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as e:
            raise CallError(f"{name}: arguments are not valid JSON ({e.msg})") from e
    if not isinstance(args, dict):
        raise CallError(f"{name}: arguments must be a JSON object")
    try:
        model = args_model(action).model_validate(args, context=ref_context(profile))
    except ValidationError as e:
        raise CallError(f"{name}: {short_errors(e)}") from e
    return Call(name=name, args=model)


def short_errors(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:4]:
        loc = ".".join(str(p) for p in err["loc"] if not str(p).startswith("function-"))
        parts.append(f"{loc or 'args'}: {err['msg']}")
    return "; ".join(parts)


def check_call(call: Call, words: list[str]) -> list[str]:
    """Problems with a call given the transcript: word indices that don't point at a
    demonstrative, labels that are not names. Empty list means fine."""
    return transcript_problems(call.name, call.args, words)


# ---------------------------------------------------------------- tools JSON


def inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline $defs/$ref and drop titles and null branches, so a schema reads cleanly in a
    prompt (a nullable field becomes an optional one)."""
    defs = schema.get("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
                extra = {k: walk(v) for k, v in node.items() if k != "$ref"}
                return {**target, **extra}
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return _drop_null_branches(walk(schema))


def _drop_null_branches(node: Any) -> Any:
    if isinstance(node, dict):
        node = {k: _drop_null_branches(v) for k, v in node.items()}
        any_of = node.get("anyOf")
        if isinstance(any_of, list) and any(b == {"type": "null"} for b in any_of):
            rest = [b for b in any_of if b != {"type": "null"}]
            node.pop("anyOf")
            if node.get("default", "x") is None:
                node.pop("default")
            if len(rest) == 1:
                node = {**rest[0], **node}
            else:
                node["anyOf"] = rest
        return node
    if isinstance(node, list):
        return [_drop_null_branches(x) for x in node]
    return node


def _prune_sizes(node: Any, presets: bool, gestures: bool) -> Any:
    """In a size argument's schema, leave out the shapes the robot can't use: presets when
    it has none, "this big" when its profile has no size gestures."""
    if isinstance(node, dict):
        node = {k: _prune_sizes(v, presets, gestures) for k, v in node.items()}
        for key in ("oneOf", "anyOf"):
            branches = node.get(key)
            if not isinstance(branches, list):
                continue
            keep = []
            for b in branches:
                props = set(b.get("properties", {})) if isinstance(b, dict) else set()
                if not presets and props == {"preset"}:
                    continue
                if not gestures and props == {"word_index", "deixis"}:
                    continue
                keep.append(b)
            if len(keep) == 1 and len(branches) > 1:
                rest = {k: v for k, v in node.items() if k not in (key, "discriminator")}
                node = {**keep[0], **rest}
            else:
                node[key] = keep
        return node
    if isinstance(node, list):
        return [_prune_sizes(x, presets, gestures) for x in node]
    return node


def tool_parameters(profile: Profile, action: Action) -> dict[str, Any]:
    """JSON Schema for one action's arguments, ready for an agent's tool list."""
    params = inline_schema(args_model(action).model_json_schema(by_alias=True))
    presets = list(profile.sizes.presets)
    if presets:
        params = with_presets(params, presets)
    props = params.get("properties", {})
    for p in action.params:
        if p.type == "size" and p.name in props:
            props[p.name] = _prune_sizes(props[p.name], bool(presets), profile.spans is not None)
    params.pop("description", None)
    params.setdefault("properties", {})
    params["type"] = "object"
    return params


def tools_for_agent(profile: Profile) -> list[dict[str, Any]]:
    """The OpenAI-format `tools` list: the robot's actions plus the standard ones."""
    return [
        {
            "type": "function",
            "function": {
                "name": a.name,
                "description": a.description or a.name,
                "parameters": tool_parameters(profile, a),
            },
        }
        for a in profile.all_actions()
        if a.agent
    ]
