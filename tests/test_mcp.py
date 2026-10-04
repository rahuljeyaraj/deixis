"""The MCP server and DeixisClient (C13, C14, plan 8.4): the same JSON as the direct call."""

import json
import statistics
import sys
import time

import pytest

from support import ROOT, profile

pytest.importorskip("mcp")

from deixis.actions import CallError, tools_for_agent  # noqa: E402
from deixis.api import RequestError, adjust, resolve  # noqa: E402
from deixis.frames import fit_transform  # noqa: E402
from deixis.mcp import DeixisClient  # noqa: E402
from deixis.mcp.client import McpUnavailable  # noqa: E402
from deixis.tracker import track  # noqa: E402
from deixis.types import Sample  # noqa: E402
from deixis.where import Size, Where  # noqa: E402

EXAMPLES = sorted((ROOT / "examples" / "requests").glob("E*.json"))


def example(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clients():
    """One server per robot profile for the whole module (each start takes about 1 s)."""
    made = {}

    def get(name):
        if name not in made:
            made[name] = DeixisClient(profile(name), mode="mcp", fallback=None)
        return made[name]

    yield get
    for c in made.values():
        c.close()


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem[:2])
def test_examples_over_mcp_equal_the_direct_call(clients, path):
    d = example(path)
    c = clients(d["profile"])
    got = c.resolve(d["request"])
    assert c.over_mcp and c.fell_back is None
    assert got.model_dump_json() == resolve(d["request"], profile(d["profile"])).model_dump_json()


def test_describe_tools_and_schemas(clients):
    c = clients("vacuum-floor")
    info = c.describe()
    assert info["package"] == "deixis" and info["schema_version"] == "1.0"
    assert info["profile"]["name"] == "vacuum-floor" and info["profile"]["frame"] == "floor"
    assert info["profile"]["actions"][:2] == ["clean", "dock"]
    assert c.tools_for_agent() == json.loads(json.dumps(tools_for_agent(profile("vacuum-floor"))))
    assert c.schema("where")["$id"].endswith("where-1.json")


def test_agent_mistakes_come_back_as_call_errors(clients):
    c = clients("arm-table")
    with pytest.raises(CallError, match="unknown tool 'fly'"):
        c.resolve({"words": [], "calls": [{"name": "fly", "args": {}}]})
    bad_index = {"object": {"last": True}, "place": {"deixis": "here", "word_index": 4}}
    with pytest.raises(RequestError, match="past the last word"):
        c.resolve({"words": [], "calls": [{"name": "move", "args": bad_index}]})
    assert c.over_mcp  # an agent's mistake is not an MCP failure


def test_adjust_track_and_frames_over_mcp(clients):
    c = clients("arm-table")
    p = profile("arm-table")
    w = Where(frame="table", kind="box", box_mm=(100, 100, 50, 50), source="gesture")
    expect = Where.model_validate_json(adjust(w, -1, 0, 1.2, p).model_dump_json())
    assert c.adjust(w, dx=-1, scale=1.2) == expect
    assert c.adjust_size(Size(mm=50, source="default"), 1.5).mm == 75
    samples = [Sample(k / 30, (300.0, 200.0)) for k in range(40)]
    assert c.track(samples) == track(samples, p)
    src = [(0, 0), (100, 0), (0, 100), (80, 90)]
    dst = [(10, 10), (10, 110), (-90, 10), (-80, 90)]
    t = c.fit_transform(src, dst, "similarity", "camera", "table")
    assert t == fit_transform(src, dst, "similarity", "camera", "table")
    moved = c.apply_transform(w.model_copy(update={"frame": "camera"}), t)
    assert moved.frame == "table" and moved.box_mm == pytest.approx((-140, 110, 50, 50))


def test_a_plain_mcp_client_can_use_the_server():
    """No DeixisClient: the official SDK talks to python -m deixis.mcp directly."""
    import asyncio

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    d = example(EXAMPLES[1])  # E2, the vacuum
    prof = ROOT / "examples" / "profiles" / f"{d['profile']}.json"

    async def go():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "deixis.mcp", "--profile", str(prof)]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                r = await s.call_tool("resolve", {"request": d["request"]})
                res = await s.list_resources()
                return names, r.structured_content, {str(x.uri) for x in res.resources}

    names, out, uris = asyncio.run(go())
    want = {"describe", "resolve", "adjust", "track", "fit_transform", "apply_transform"}
    assert want | {"tools_for_agent", "schema", "set_profile"} <= names
    assert out["status"] == "ok"
    assert out["calls"][0]["args"]["place"]["box_mm"] == [1020.0, 480.0, 1000.0, 1000.0]
    assert "deixis://schema/where/1.0" in uris


def test_record_writes_every_call(tmp_path):
    d = example(EXAMPLES[0])
    with DeixisClient(profile(d["profile"]), mode="mcp", fallback=None, record=tmp_path) as c:
        c.resolve(d["request"])
        with pytest.raises(CallError):
            c.resolve({"calls": [{"name": "nope"}]})
    lines = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [x["tool"] for x in lines] == ["set_profile", "resolve", "resolve"]
    assert lines[1]["result"]["status"] == "ok" and lines[1]["args"]["calls"][0]["name"]
    assert lines[2]["error"].startswith("CallError")


def test_fallback_to_direct_when_the_server_cannot_start(caplog):
    d = example(EXAMPLES[3])
    dead = [sys.executable, "-c", "pass"]
    c = DeixisClient(profile(d["profile"]), mode="mcp", command=dead, start_timeout=10)
    assert not c.over_mcp and "could not start" in c.fell_back
    assert "falling back" in caplog.text
    assert c.resolve(d["request"]).status == "ok"
    with pytest.raises(McpUnavailable):
        DeixisClient(profile(d["profile"]), mode="mcp", fallback=None, command=dead,
                     start_timeout=10)  # fmt: skip


def test_fallback_when_the_server_dies():
    d = example(EXAMPLES[3])
    c = DeixisClient(profile(d["profile"]), mode="mcp")
    assert c.over_mcp
    c._conn.close()  # the server goes away mid-session
    assert c.resolve(d["request"]).status == "ok"
    assert not c.over_mcp and "resolve" in c.fell_back


def test_direct_mode_and_the_environment(monkeypatch):
    d = example(EXAMPLES[0])
    monkeypatch.setenv("DEIXIS_MODE", "direct")
    c = DeixisClient(profile(d["profile"]))
    assert c.mode == "direct" and c._conn is None and not c.over_mcp
    assert c.resolve(d["request"]).status == "ok" and c.calls == 0
    monkeypatch.setenv("DEIXIS_MODE", "sideways")
    with pytest.raises(ValueError, match="mode"):
        DeixisClient(profile(d["profile"]))


def big_request(n=900):
    """About 30 s of fingertip samples at 30 fps, and one call."""
    samples = []
    for k in range(n):
        t = k / 30
        x = 100 + 40 * (t // 3 % 2) + 0.2 * ((k * 7) % 5)
        samples.append({"t": round(t, 4), "point": [x, 80.0 + 0.1 * (k % 3)]})
    words = [{"text": "draw", "t0": 28.0, "t1": 28.3}, {"text": "here", "t0": 28.3, "t1": 28.6}]
    place = {"deixis": "here", "word_index": 1}
    calls = [{"name": "place_drawing", "args": {"object": "cat", "place": place}}]
    return {"words": words, "samples": samples, "calls": calls}


def test_latency_of_a_900_sample_request(clients):
    """Plan 8.4: target median below 30 ms for a 900-sample request (measured by
    scripts/latency.py and written in LIMITATIONS.md L15)."""
    c = clients("plotter-a5")
    req = big_request()
    c.resolve(req)  # warm up
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = c.resolve(req)
        times.append((time.perf_counter() - t0) * 1000)
    assert r.status == "ok"
    median = statistics.median(times)
    print(f"\n900-sample resolve over MCP: median {median:.1f} ms, max {max(times):.1f} ms")
    assert median < 100  # generous here: test machines vary; the target is checked by the script
