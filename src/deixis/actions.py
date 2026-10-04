"""What a robot can do: its actions and their parameters, declared in its profile.

The core uses them to know each argument's type when resolving (a `place`, an `object`, a
`size`, or something it only passes through), to build the tool list for any agent, and to
validate the agent's calls. The robot decides what each action does.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def _unique(self) -> Action:
        names = [p.name for p in self.params]
        if len(names) != len(set(names)):
            raise ValueError(f"action '{self.name}': parameter names must be unique")
        return self

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


#: Actions every profile gets. They need no robot code to resolve; the robot decides what
#: they do.
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
)
