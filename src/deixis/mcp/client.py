"""`DeixisClient`: a small synchronous client for robots (plan 4.9).

The robot apps are threaded, not async. The client starts `deixis-mcp` as a child process,
runs the async MCP session in a background thread, and offers plain methods: `resolve`,
`adjust`, `describe`, `track`, `fit_transform`, `apply_transform`, `tools_for_agent`,
`schema`. Each call has a timeout (2 s by default).

Two modes, one setting (`mode=`, or the environment variable `DEIXIS_MODE=mcp|direct`):

- `mcp`: every call goes to the server. With `fallback="direct"` (the default), a failed
  call (the server died, a timeout) is done in-process instead, logged loudly, and the
  client stays in-process from then on (`fell_back` says why).
- `direct`: every call runs `deixis` in-process; the MCP SDK is not needed.

Mistakes the agent must fix (`CallError`) and malformed requests (`RequestError`) are raised
the same way in both modes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Literal, TypeVar

from deixis.actions import CallError, tools_for_agent
from deixis.api import Request, RequestError, Response, adjust, adjust_size, resolve
from deixis.frames import FitError, Transform, fit_transform
from deixis.profile import Profile, load_profile
from deixis.resolver import SpanHold
from deixis.schema import schemas
from deixis.tracker import track
from deixis.types import Hold, Sample
from deixis.where import Size, Where

log = logging.getLogger("deixis.mcp")

Mode = Literal["mcp", "direct"]
T = TypeVar("T")

_ERRORS: dict[str, type[Exception]] = {
    "CallError": CallError,
    "RequestError": RequestError,
    "FitError": FitError,
    "ValueError": ValueError,
}


class McpUnavailable(RuntimeError):
    """The MCP server could not be reached (and there is no fallback)."""


def server_command(record: str | Path | None = None) -> list[str]:
    """The command that starts `deixis-mcp` with this Python (no profile: it is sent)."""
    cmd = [sys.executable, "-m", "deixis.mcp"]
    if record is not None:
        cmd += ["--record", str(record)]
    return cmd


class _Connection:
    """One `deixis-mcp` child process and its MCP session, in a background thread."""

    def __init__(self, command: Sequence[str], start_timeout: float) -> None:
        from mcp.client.stdio import StdioServerParameters

        self.params = StdioServerParameters(command=command[0], args=list(command[1:]))
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.session: Any = None
        self.closing: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._run, name="deixis-mcp", daemon=True)
        self.thread.start()
        if not self.ready.wait(start_timeout):
            raise McpUnavailable(f"deixis-mcp did not start within {start_timeout} s")
        if self.error is not None:
            raise McpUnavailable(f"deixis-mcp did not start: {self.error!r}") from self.error

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main())
        finally:
            self.loop.close()

    async def _main(self) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        self.closing = asyncio.Event()
        try:
            async with stdio_client(self.params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.ready.set()
                    await self.closing.wait()
        except BaseException as e:  # noqa: BLE001 - reported to the caller
            self.error = e
        finally:
            self.session = None
            self.ready.set()

    @property
    def alive(self) -> bool:
        return self.session is not None and self.thread.is_alive()

    def call(self, tool: str, args: dict[str, Any], timeout: float) -> Any:
        if not self.alive:
            raise McpUnavailable(f"deixis-mcp is not running ({self.error!r})")
        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(tool, args), self.loop)
        try:
            return fut.result(timeout)
        except FutureTimeout:
            fut.cancel()
            raise McpUnavailable(f"deixis-mcp did not answer {tool} within {timeout} s") from None

    def close(self) -> None:
        if self.closing is not None and self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.closing.set)
        self.thread.join(10)


class DeixisClient:
    """Synchronous access to the core, over MCP or in-process."""

    def __init__(
        self,
        profile: str | Path | dict[str, Any] | Profile,
        mode: Mode | None = None,
        fallback: Literal["direct"] | None = "direct",
        timeout: float = 2.0,
        command: Sequence[str] | None = None,
        record: str | Path | None = None,
        start_timeout: float = 30.0,
    ) -> None:
        self.profile = load_profile(profile)
        mode = mode or os.environ.get("DEIXIS_MODE", "mcp")  # type: ignore[assignment]
        if mode not in ("mcp", "direct"):
            raise ValueError(f"mode must be 'mcp' or 'direct', not {mode!r}")
        self.mode: Mode = mode  # type: ignore[assignment]
        self.fallback = fallback
        self.timeout = timeout
        self.fell_back: str | None = None  # why the client went in-process, if it did
        self.calls = 0  # calls made over MCP
        self._lock = threading.Lock()
        self._conn: _Connection | None = None
        if self.mode == "mcp":
            try:
                self._conn = _Connection(command or server_command(record), start_timeout)
                self._mcp("set_profile", {"profile": self.profile.model_dump(mode="json")})
            except (McpUnavailable, ImportError) as e:
                self._give_up(f"could not start deixis-mcp: {e}")

    # ------------------------------------------------------------------ plumbing

    @property
    def over_mcp(self) -> bool:
        """True while calls really go over MCP."""
        return self.mode == "mcp" and self.fell_back is None

    def _give_up(self, why: str) -> None:
        if self.fallback != "direct":
            raise McpUnavailable(why)
        log.error("deixis: MCP failed, falling back to in-process calls from now on: %s", why)
        self.fell_back = why
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _mcp(self, tool: str, args: dict[str, Any]) -> Any:
        from mcp import MCPError

        assert self._conn is not None
        with self._lock:
            try:
                result = self._conn.call(tool, args, self.timeout)
            except MCPError as e:
                data = e.data if isinstance(e.data, dict) else {}
                err = _ERRORS.get(str(data.get("type")))
                if err is not None:
                    raise err(e.message) from None
                raise McpUnavailable(f"{tool}: MCP error {e.code}: {e.message}") from e
            self.calls += 1
        if getattr(result, "is_error", False):
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise McpUnavailable(f"{tool}: {text or 'tool error'}")
        data = result.structured_content
        if isinstance(data, dict) and set(data) == {"result"}:
            return data["result"]
        return data

    def _run(self, tool: str, args: Callable[[], dict[str, Any]], direct: Callable[[], T],
             parse: Callable[[Any], T]) -> T:  # fmt: skip
        if not self.over_mcp:
            return direct()
        try:
            return parse(self._mcp(tool, args()))
        except (CallError, RequestError, FitError):
            raise
        except ValueError as e:
            raise ValueError(str(e)) from None
        except Exception as e:  # noqa: BLE001 - the server died or did not answer
            self._give_up(f"{tool}: {e!r}")
            return direct()

    # ------------------------------------------------------------------ the core's API

    def resolve(self, request: Request | dict[str, Any]) -> Response:
        """`deixis.resolve` for this robot's profile."""
        req = request if isinstance(request, Request) else Request.model_validate(request)
        return self._run(
            "resolve",
            lambda: {"request": req.model_dump(mode="json")},
            lambda: resolve(req, self.profile),
            Response.model_validate,
        )

    def adjust(self, where: Where, dx: float = 0, dy: float = 0, scale: float = 1.0) -> Where:
        args = {"where": where.model_dump(mode="json"), "dx": dx, "dy": dy, "scale": scale}
        return self._run(
            "adjust",
            lambda: args,
            lambda: adjust(where, dx, dy, scale, self.profile),
            lambda d: Where.model_validate(d["where"]),
        )

    def adjust_size(self, size: Size, scale: float) -> Size:
        return self._run(
            "adjust",
            lambda: {"size": size.model_dump(mode="json"), "scale": scale},
            lambda: adjust_size(size, scale),
            lambda d: Size.model_validate(d["size"]),
        )

    def describe(self) -> dict[str, Any]:
        def direct() -> dict[str, Any]:
            import deixis
            from deixis.where import WIRE_VERSION

            p = self.profile
            return {
                "package": "deixis",
                "version": deixis.__version__,
                "schema_version": WIRE_VERSION,
                "profile": {
                    "name": p.name,
                    "frame": p.frame,
                    "area_mm": list(p.area_mm),
                    "view": p.view,
                    "actions": [a.name for a in p.all_actions() if a.agent],
                },
            }

        return self._run("describe", dict, direct, dict)

    def track(self, samples: Sequence[Sample]) -> tuple[list[Hold], list[SpanHold]]:
        from pydantic import TypeAdapter

        def parse(d: Any) -> tuple[list[Hold], list[SpanHold]]:
            holds = TypeAdapter(list[Hold]).validate_python(d["holds"])
            spans = TypeAdapter(list[SpanHold]).validate_python(d["spans"])
            return holds, spans

        return self._run(
            "track",
            lambda: {"samples": TypeAdapter(list[Sample]).dump_python(list(samples), mode="json")},
            lambda: track(samples, self.profile),
            parse,
        )

    def fit_transform(
        self,
        src: Sequence[Sequence[float]],
        dst: Sequence[Sequence[float]],
        kind: Literal["similarity", "homography"] = "similarity",
        from_frame: str | None = None,
        to_frame: str | None = None,
        allow_mirror: bool = False,
    ) -> Transform:
        args = {
            "src": [[float(a), float(b)] for a, b in src],
            "dst": [[float(a), float(b)] for a, b in dst],
            "kind": kind,
            "from_frame": from_frame,
            "to_frame": to_frame,
            "allow_mirror": allow_mirror,
        }
        return self._run(
            "fit_transform",
            lambda: args,
            lambda: fit_transform(src, dst, kind, from_frame, to_frame, allow_mirror),
            Transform.model_validate,
        )

    def apply_transform(
        self, where: Where, transform: Transform, to_frame: str | None = None
    ) -> Where:
        return self._run(
            "apply_transform",
            lambda: {
                "where": where.model_dump(mode="json"),
                "transform": transform.model_dump(mode="json"),
                "to_frame": to_frame,
            },
            lambda: transform.apply_where(where, to_frame),
            Where.model_validate,
        )

    def tools_for_agent(self) -> list[dict[str, Any]]:
        return self._run("tools_for_agent", dict, lambda: tools_for_agent(self.profile), list)

    def schema(self, name: str) -> dict[str, Any]:
        return self._run("schema", lambda: {"name": name}, lambda: schemas()[name], dict)

    # ------------------------------------------------------------------ lifetime

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DeixisClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
