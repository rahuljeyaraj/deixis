"""Helpers shared by the tests (importable: pytest puts tests/ on the path)."""

from pathlib import Path

from deixis.profile import Profile, load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "examples" / "profiles"
PROFILE_NAMES = ("plotter-a5", "vacuum-floor", "arm-table")


def profile(name: str) -> Profile:
    return load_profile(PROFILES / f"{name}.json")
