"""`deixis-mcp`: the core as an MCP server over stdio (plan 4.9).

One server process per robot, started by the robot's app:

    deixis-mcp --profile path/to/profile.json [--record out/dir]

It is stateless apart from the profile (given at start, or with `set_profile`). Pointer
samples run at 30 frames a second, but the core needs them once per utterance: the robot
keeps a short buffer and sends one `resolve` per utterance (about 900 samples, 60 kB).

Errors the agent must fix (a bad tool call) and malformed requests come back as MCP errors
whose `data.type` is `CallError` or `RequestError`, with the message for the model.

With `--record`, every call and its result are appended to `calls.jsonl` in that folder:
they become replay tests. This module may use the clock (for the record); the core may not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import MCPError
from mcp.server.mcpserver import MCPServer

import deixis
from deixis.actions import CallError, tools_for_agent
from deixis.api import Request, RequestError, Response, adjust, adjust_size, resolve
from deixis.frames import FitError, Transform, fit_transform
from deixis.profile import Profile, load_profile
from deixis.schema import schemas
from deixis.tracker import track
from deixis.types import Sample
from deixis.where import WIRE_VERSION, Size, Where

INVALID_PARAMS = -32602


class _State:
    def __init__(self, profile: Profile | None, record: Path | None) -> None:
        self.profile = profile
        self.record = record
        self.lock = threading.Lock()
        self.seq = 0
        if record is not None:
            record.mkdir(parents=True, exist_ok=True)

    def need_profile(self) -> Profile:
        if self.profile is None:
            raise MCPError(
                INVALID_PARAMS,
                "no profile: start with --profile or call set_profile first",
                {"type": "RequestError"},
            )
        return self.profile

    def log(self, tool: str, args: dict[str, Any], result: Any, error: str | None = None) -> None:
        if self.record is None:
            return
        with self.lock:
            self.seq += 1
            entry = {
                "seq": self.seq,
                "at": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
                "tool": tool,
                "profile": self.profile.name if self.profile else None,
                "args": args,
                "result": result,
            }
            if error is not None:
                entry["error"] = error
            with (self.record / "calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _error(e: Exception) -> MCPError:
    return MCPError(INVALID_PARAMS, str(e), {"type": type(e).__name__})


def build_server(profile: Profile | None = None, record: Path | None = None) -> MCPServer:
    """The MCP server with the tools of plan 4.9."""
    state = _State(profile, record)
    mcp = MCPServer(
        "deixis",
        version=deixis.__version__,
        instructions=(
            "Turns 'here', 'that one' and 'this big', plus pointing, into places a robot can "
            "act on. Call resolve once per utterance with the words, pointer samples or holds, "
            "the things in the scene and the agent's tool calls (references, no coordinates)."
        ),
    )

    @mcp.tool()
    def describe() -> dict[str, Any]:
        """The package and schema versions, and the profile: name, frame, area, actions."""
        p = state.profile
        out: dict[str, Any] = {
            "package": "deixis",
            "version": deixis.__version__,
            "schema_version": WIRE_VERSION,
            "profile": None,
        }
        if p is not None:
            out["profile"] = {
                "name": p.name,
                "frame": p.frame,
                "area_mm": list(p.area_mm),
                "view": p.view,
                "actions": [a.name for a in p.all_actions() if a.agent],
            }
        return out

    @mcp.tool()
    def set_profile(profile: Profile) -> dict[str, Any]:
        """Use this robot profile from now on."""
        state.profile = profile
        state.log("set_profile", {"name": profile.name}, {"ok": True})
        return {"ok": True, "name": profile.name, "frame": profile.frame}

    @mcp.tool(name="resolve")
    def resolve_tool(request: Request) -> Response:
        """Resolve every reference in the request's tool calls: each place, object and size
        becomes a Where or a Size, or the response asks one question (status 'ask')."""
        p = state.need_profile()
        args = request.model_dump(mode="json")
        try:
            response = resolve(request, p)
        except (CallError, RequestError) as e:
            state.log("resolve", args, None, f"{type(e).__name__}: {e}")
            raise _error(e) from e
        state.log("resolve", args, response.model_dump(mode="json"))
        return response

    @mcp.tool(name="adjust")
    def adjust_tool(
        where: Where | None = None,
        size: Size | None = None,
        dx: float = 0,
        dy: float = 0,
        scale: float = 1.0,
    ) -> dict[str, Any]:
        """A small fix: move a Where by dx, dy steps (the person's right and away) and scale
        it, or scale a Size. Returns {"where": ..., "size": ...} for what was given."""
        p = state.need_profile()
        out: dict[str, Any] = {}
        if where is not None:
            out["where"] = adjust(where, dx, dy, scale, p).model_dump(mode="json")
        if size is not None:
            out["size"] = adjust_size(size, scale).model_dump(mode="json")
        state.log("adjust", {"dx": dx, "dy": dy, "scale": scale}, out)
        return out

    @mcp.tool(name="track")
    def track_tool(samples: list[Sample]) -> dict[str, Any]:
        """Holds and size gestures from pointer samples (with the profile's rules)."""
        p = state.need_profile()
        holds, spans = track(samples, p)
        return {
            "holds": [_dc(h) for h in holds],
            "spans": [_dc(s) for s in spans],
        }

    @mcp.tool(name="fit_transform")
    def fit_transform_tool(
        src: list[tuple[float, float]],
        dst: list[tuple[float, float]],
        kind: str = "similarity",
        from_frame: str | None = None,
        to_frame: str | None = None,
        allow_mirror: bool = False,
    ) -> Transform:
        """Fit a similarity (2 or more point pairs) or homography (4 or more) from src to dst;
        the result has its residuals (rms_mm, max_mm)."""
        try:
            t = fit_transform(src, dst, kind, from_frame, to_frame, allow_mirror)  # type: ignore[arg-type]
        except (FitError, ValueError) as e:
            raise _error(e) from e
        state.log("fit_transform", {"pairs": len(src), "kind": kind}, t.model_dump(mode="json"))
        return t

    @mcp.tool(name="apply_transform")
    def apply_transform_tool(
        where: Where, transform: Transform, to_frame: str | None = None
    ) -> Where:
        """The same Where in another frame: point, box corners, path and covariance."""
        try:
            return transform.apply_where(where, to_frame)
        except ValueError as e:
            raise _error(e) from e

    @mcp.tool(name="tools_for_agent")
    def tools_for_agent_tool() -> list[dict[str, Any]]:
        """The OpenAI-format tool list for this profile (its actions and the standard ones)."""
        return tools_for_agent(state.need_profile())

    @mcp.tool(name="schema")
    def schema_tool(name: str) -> dict[str, Any]:
        """The JSON Schema of a wire type: where, size, ask, answer, thing, transform, request,
        response or profile."""
        all_ = schemas()
        if name not in all_:
            raise MCPError(INVALID_PARAMS, f"no schema '{name}'; schemas: {', '.join(all_)}")
        return all_[name]

    for name in schemas():
        _schema_resource(mcp, name)
    return mcp


def _schema_resource(mcp: MCPServer, name: str) -> None:
    @mcp.resource(
        f"deixis://schema/{name}/{WIRE_VERSION}",
        name=f"{name}-schema",
        description=f"JSON Schema of the {name} message, version {WIRE_VERSION}.",
        mime_type="application/schema+json",
    )
    def read() -> str:
        return json.dumps(schemas()[name], indent=2)


def _dc(x: Any) -> dict[str, Any]:
    """A dataclass (Hold, SpanHold) as JSON-ready data."""
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(x)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deixis-mcp", description=__doc__.splitlines()[0])
    ap.add_argument("--profile", type=Path, help="the robot's profile (JSON)")
    ap.add_argument("--record", type=Path, help="append every call to calls.jsonl here")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    profile = load_profile(args.profile) if args.profile else None
    build_server(profile, args.record).run("stdio")
    return 0
