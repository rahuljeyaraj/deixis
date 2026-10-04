"""Replay cases (plan 8.2, C16): requests the plotter really sent, in its virtual scenes and
on the Ender on 2 Oct, with the answers it got. Exported by `deixis-doodle`'s
`scripts/export_deixis_cases.py`. Each must resolve the same today (numbers within 0.01 mm,
everything else equal), in-process and over MCP (identical JSON)."""

import json
from pathlib import Path

import pytest

from deixis.api import resolve
from deixis.profile import load_profile

HERE = Path(__file__).resolve().parent / "replay"
CASES = sorted((HERE / "cases").glob("*.json"))


def case(path):
    return json.loads(path.read_text(encoding="utf-8"))


def profile_of(c):
    return load_profile(HERE / "profiles" / f"{c['profile']}.json")


def same(got, want, where="response"):
    """Equal, but numbers only within 0.01 (mm, s)."""
    if isinstance(want, bool) or want is None or isinstance(want, str):
        assert got == want, f"{where}: {got!r} != {want!r}"
    elif isinstance(want, (int, float)):
        assert isinstance(got, (int, float)) and abs(got - want) <= 0.01, (
            f"{where}: {got} != {want}"
        )
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{where}: {got!r} != {want!r}"
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            same(g, w, f"{where}[{k}]")
    else:
        assert isinstance(got, dict) and set(got) == set(want), f"{where}: keys differ"
        for k in want:
            same(got[k], want[k], f"{where}.{k}")


def test_there_are_enough_cases():
    sources = {case(p)["source"] for p in CASES}
    assert len(CASES) >= 40 and any(s.startswith("live-") for s in sources)


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_replay(path):
    c = case(path)
    got = resolve(c["request"], profile_of(c))
    same(got.model_dump(mode="json"), c["response"])


def test_replay_over_mcp_is_identical():
    pytest.importorskip("mcp")
    from deixis.mcp import DeixisClient

    clients = {}
    try:
        for path in CASES:
            c = case(path)
            p = profile_of(c)
            if c["profile"] not in clients:
                clients[c["profile"]] = DeixisClient(p, mode="mcp", fallback=None)
            client = clients[c["profile"]]
            got = client.resolve(c["request"]).model_dump_json()
            assert got == resolve(c["request"], p).model_dump_json(), path.stem
            assert client.over_mcp
    finally:
        for client in clients.values():
            client.close()
