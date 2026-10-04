"""deixis: turn "here", "that one" and "this big", plus pointing, into a place a robot can
act on.

Robot-agnostic and pure: no models, camera, network or robot; the same input gives the same
output. Everything robot-specific comes from the robot's `Profile`.
"""

from deixis.actions import Action, Call, CallError, Param, check_call, parse_call, tools_for_agent
from deixis.api import Request, Response, adjust, adjust_size, resolve
from deixis.frames import Transform, fit_transform
from deixis.geometry import Box, Point, Polyline
from deixis.profile import Profile, load_profile
from deixis.scene import Thing
from deixis.schema import schemas
from deixis.tracker import Tracker, track
from deixis.types import Hold, Sample, Word
from deixis.where import Answer, Ask, Size, Where

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Call",
    "CallError",
    "Param",
    "Answer",
    "Ask",
    "Box",
    "Hold",
    "Point",
    "Polyline",
    "Profile",
    "Request",
    "Response",
    "Sample",
    "Size",
    "Thing",
    "Tracker",
    "Transform",
    "Where",
    "Word",
    "__version__",
    "adjust",
    "adjust_size",
    "check_call",
    "fit_transform",
    "load_profile",
    "parse_call",
    "resolve",
    "schemas",
    "tools_for_agent",
    "track",
]
