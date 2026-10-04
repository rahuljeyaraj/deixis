"""Time per resolve over MCP and in-process, for LIMITATIONS.md L15 (plan C14, 8.4).

Run: uv run python scripts/latency.py [--n 200]

Sends a request with about 30 s of fingertip samples (900 at 30 fps) and one call, through
DeixisClient in MCP mode (a real deixis-mcp child process over stdio) and in direct mode,
and prints the median, 95th percentile and worst time. Target: median below 30 ms over MCP.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from deixis.mcp import DeixisClient  # noqa: E402
from test_mcp import big_request  # noqa: E402

PROFILE = ROOT / "examples" / "profiles" / "plotter-a5.json"


def measure(client: DeixisClient, request: dict, n: int) -> list[float]:
    client.resolve(request)  # warm up
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        client.resolve(request)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def report(name: str, ms: list[float]) -> str:
    ms = sorted(ms)
    p95 = ms[int(0.95 * (len(ms) - 1))]
    return f"{name}: median {statistics.median(ms):.1f} ms, p95 {p95:.1f} ms, max {ms[-1]:.1f} ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--samples", type=int, default=900)
    args = ap.parse_args()
    req = big_request(args.samples)
    print(f"{platform.platform()}, Python {platform.python_version()}, {args.samples} samples")
    t0 = time.perf_counter()
    with DeixisClient(PROFILE, mode="mcp", fallback=None) as c:
        print(f"server start and profile: {(time.perf_counter() - t0) * 1000:.0f} ms")
        print(report("over MCP", measure(c, req, args.n)))
    with DeixisClient(PROFILE, mode="direct") as c:
        print(report("in-process", measure(c, req, args.n)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
