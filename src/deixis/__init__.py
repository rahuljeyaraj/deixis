"""deixis: turn "here", "that one" and "this big", plus pointing, into a place a robot can
act on.

Robot-agnostic and pure: no models, camera, network or robot; the same input gives the same
output. Everything robot-specific comes from the robot's `Profile`.
"""

from deixis.geometry import Box, Point, Polyline
from deixis.profile import Profile, load_profile
from deixis.schema import schemas
from deixis.types import Hold, Word
from deixis.where import Answer, Ask, Size, Where

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Ask",
    "Box",
    "Hold",
    "Point",
    "Polyline",
    "Profile",
    "Size",
    "Where",
    "Word",
    "__version__",
    "load_profile",
    "schemas",
]
