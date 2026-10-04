"""Helpers shared by the tests (importable: pytest puts tests/ on the path)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deixis.api import Request, Response, resolve
from deixis.profile import Profile, load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "examples" / "profiles"
PROFILE_NAMES = ("plotter-a5", "vacuum-floor", "arm-table")


def profile(name: str) -> Profile:
    return load_profile(PROFILES / f"{name}.json")


def with_actions(p: Profile, *actions: dict[str, Any], **changes: Any) -> Profile:
    """The profile with extra actions (and other top-level changes), validated again."""
    d = p.model_dump(mode="json")
    d["actions"] = [*d["actions"], *actions]
    d.update(changes)
    return Profile.model_validate(d)


def words(*spec: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [{"text": t, "t0": a, "t1": b} for t, a, b in spec]


def say(text: str, t0: float = 10.0, step: float = 0.3) -> list[dict[str, Any]]:
    """Words of a sentence, one every `step` seconds from t0."""
    return [
        {"text": w, "t0": round(t0 + k * step, 3), "t1": round(t0 + (k + 1) * step, 3)}
        for k, w in enumerate(text.split())
    ]


def hold(
    t0: float, t1: float, x: float, y: float, spread: float = 0.5, **kw: Any
) -> dict[str, Any]:
    return {"t0": t0, "t1": t1, "mean": [x, y], "spread": [spread, spread], **kw}


def thing(id: str, box: tuple[float, float, float, float], **kw: Any) -> dict[str, Any]:
    return {"id": id, "label": kw.pop("label", id), "box_mm": list(box), **kw}


def call(name: str, **args: Any) -> dict[str, Any]:
    if "from_" in args:
        args["from"] = args.pop("from_")
    return {"name": name, "args": args}


def run(p: Profile, **req: Any) -> Response:
    """resolve(), checking that the same request gives byte-identical JSON twice."""
    request = Request.model_validate(req)
    a = resolve(request, p)
    b = resolve(Request.model_validate_json(request.model_dump_json()), p)
    assert a.model_dump_json() == b.model_dump_json()
    return a


def scale(p: Profile, fx: float, fy: float) -> tuple[float, float]:
    """A point at a fraction of the profile's work area."""
    a = p.area
    return (round(a.x + fx * a.w, 2), round(a.y + fy * a.h, 2))
