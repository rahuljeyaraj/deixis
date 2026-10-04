"""Shared fixtures: the three example profiles (plotter, vacuum, arm)."""

import pytest

from deixis.profile import Profile
from support import profile


@pytest.fixture
def plotter() -> Profile:
    return profile("plotter-a5")


@pytest.fixture
def vacuum() -> Profile:
    return profile("vacuum-floor")


@pytest.fixture
def arm() -> Profile:
    return profile("arm-table")
