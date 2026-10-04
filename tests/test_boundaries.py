"""The core's boundaries (plan 1.3, 8.4): no robot words, pure, small dependencies."""

import re
import subprocess
import sys

from support import ROOT

SRC = ROOT / "src" / "deixis"
ROBOT_WORDS = re.compile(
    r"\b(paper|plotter|pen|vacuum|ecovacs|deebot|floor|table|arm|so-?101|mujoco|draw|clean"
    r"|gcode|bed)\b",
    re.IGNORECASE,
)
CLOCKS = re.compile(r"\btime\.time\b|\bdatetime\.now\b|\bimport random\b|\brandom\.")


def _core_files():
    return sorted(p for p in SRC.rglob("*.py") if "mcp" not in p.relative_to(SRC).parts)


def test_no_robot_words_in_the_core():
    """Frame names, sizes, question texts and actions come from the profile, never the code."""
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in ROBOT_WORDS.finditer(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {m.group(0)!r} in {line.strip()}")
    assert not hits, "robot words in src/deixis:\n" + "\n".join(hits)


def test_no_clocks_or_randomness_in_the_core():
    hits = []
    for path in _core_files():
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CLOCKS.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_import_deixis_pulls_in_nothing_heavy():
    code = (
        "import sys, deixis, deixis.api, deixis.scene, deixis.frames, deixis.tracker\n"
        "bad = {'cv2', 'mediapipe', 'openai', 'requests', 'serial', 'torch', 'mcp', 'anyio'}\n"
        "print(sorted(bad & {m.split('.')[0] for m in sys.modules}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_only_pydantic_and_numpy_are_required():
    import tomllib

    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {re.split(r"[<>=!~\[ ]", d)[0] for d in meta["project"]["dependencies"]}
    assert deps == {"pydantic", "numpy"}
