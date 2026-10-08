"""Things in the scene, and places relative to them (plan 4.7, F3, F4, F8, F9).

Everything is in the robot's frame, in mm. The robot side tells the core what is there
(`Thing`: id, label, box, attributes, when it was added, height); the core never sees.

- **That one:** the selectable thing nearest a pointed spot (0 = inside its box); two about
  equally near: ask.
- **By name:** a label or attribute ("the house", "the blue one"), most recent first.
- **Relations** (next to, left of, right of, above, below, behind, in front of, between,
  inside, on) and **named regions** ("top left corner") are computed by geometry here,
  never by a language model (LR 3.2). How "above" and "on" read depends on the profile's
  view mode, and "left" and "away" on its viewer directions.
- **Empty space:** the centre of the largest empty square on a grid over the work area,
  every thing's box kept clear by a margin (NumPy; repeated 3x3 erosion is exactly the
  chessboard distance).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from pydantic import Field

from deixis.geometry import Box, Point
from deixis.where import Box4, Wire

if TYPE_CHECKING:
    from deixis.profile import Profile

#: "on" in page view: the new thing's centre sits this many half-sizes above the target's
#: top edge (so it overlaps the edge a little, like a hat on a head).
ON_LIFT = 0.7

#: Words that say nothing about which thing is meant ("the blue one").
_FILLERS = {"one", "ones", "thing", "things", "object", "objects", "this", "that", "these", "those"}
_ARTICLES = ("the ", "a ", "an ", "my ", "your ")


class Thing(Wire):
    """Something in the scene, reported by the robot side."""

    id: str = Field(min_length=1, description="Unique, chosen by the robot side.")
    label: str | None = Field(default=None, description="What it is: 'house', 'red block'.")
    box_mm: Box4 = Field(description="Its footprint in the frame: x, y, w, h.")
    kind: str = Field(default="object", description="Free text for the robot.")
    attrs: dict[str, str] = Field(default_factory=dict, description='e.g. {"color": "blue"}.')
    t: float = Field(default=0.0, description="When it was added or last moved, s.")
    height_mm: float | None = Field(default=None, description="Its height, for 'on that'.")
    selectable: bool = Field(
        default=True, description="false: pointing and 'it' skip it (text labels, arrows)."
    )

    @property
    def box(self) -> Box:
        return Box(*self.box_mm)

    @property
    def center(self) -> Point:
        return self.box.center


class Spot(NamedTuple):
    """A place computed from things: its centre, and the height when stacked ('on')."""

    center: Point
    z_mm: float | None = None


class Unsupported(ValueError):
    """The relation has no meaning in this view mode ('below' a block on a surface)."""


# ---------------------------------------------------------------- boxes


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x, b.x))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def union_box(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x2 for b in boxes)
    y1 = max(b.y2 for b in boxes)
    return Box(x0, y0, x1 - x0, y1 - y0)


def overlap(a: Box, b: Box) -> bool:
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2


def distance_to_box(p: Point, box: Box) -> float:
    """0 inside the box, else the distance to its nearest edge."""
    dx = max(box.x - p[0], 0.0, p[0] - box.x2)
    dy = max(box.y - p[1], 0.0, p[1] - box.y2)
    return math.hypot(dx, dy)


def nearest(p: Point, points: Sequence[Point]) -> tuple[int, float] | None:
    """Index of the point nearest p, and its distance (the first one on a tie)."""
    if not points:
        return None
    i = min(range(len(points)), key=lambda k: math.hypot(points[k][0] - p[0], points[k][1] - p[1]))
    return i, math.hypot(points[i][0] - p[0], points[i][1] - p[1])


# ---------------------------------------------------------------- finding things


def _norm(s: str) -> str:
    s = s.strip().lower()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art) :]
    return s.strip(" '\"")


def _same(a: str, b: str) -> bool:
    return a == b or a.rstrip("s") == b.rstrip("s")


def _words(s: str) -> set[str]:
    return {w.rstrip("s") for w in _norm(s).replace("_", " ").replace("-", " ").split() if w}


def matches(thing: Thing, label: str) -> bool:
    """True if `label` names this thing: its label or an attribute exactly, or every word
    of `label` (apart from 'one', 'thing') among its label's words and attribute values."""
    return _exact(thing, _norm(label)) or _loose(thing, label)


def _exact(thing: Thing, want: str) -> bool:
    if thing.label and _same(_norm(thing.label), want):
        return True
    return any(_norm(v) == want for v in thing.attrs.values())


def _loose(thing: Thing, label: str) -> bool:
    want = _words(label) - {w.rstrip("s") for w in _FILLERS}
    if not want:
        return False
    have = _words(thing.label or "")
    for v in thing.attrs.values():
        have |= _words(v)
    return want <= have


def find(things: Sequence[Thing], label: str) -> list[Thing]:
    """Things with this label or attribute, most recent first.

    Exact names first ("house" is a house, or houses); only if there are none, things whose
    label and attributes contain every word said ("the blue one" -> color blue)."""
    want = _norm(label)
    hits = [o for o in things if _exact(o, want)]
    if not hits:
        hits = [o for o in things if _loose(o, label)]
    return sorted(hits, key=lambda o: o.t, reverse=True)


def last(things: Sequence[Thing], last_id: str | None = None) -> Thing | None:
    """'It': the thing the robot names as the last one, else the newest selectable thing."""
    if last_id is not None:
        hit = next((o for o in things if o.id == last_id), None)
        if hit is not None:
            return hit
    cands = [o for o in things if o.selectable]
    return max(cands, key=lambda o: o.t, default=None)


def labels(things: Sequence[Thing]) -> list[str]:
    """The selectable things' labels, oldest first, without repeats (for an agent or an Ask)."""
    out: list[str] = []
    for o in sorted(things, key=lambda o: o.t):
        if o.selectable and o.label and o.label not in out:
            out.append(o.label)
    return out


def near(
    things: Sequence[Thing], p: Point, profile: Profile, max_mm: float | None = None
) -> list[tuple[Thing, float]]:
    """Selectable things near a point (0 = inside the box), nearest first; on a tie the
    smaller box first."""
    lim = profile.scene.near_mm if max_mm is None else max_mm
    out = [(o, distance_to_box(p, o.box)) for o in things if o.selectable]
    out = [(o, d) for o, d in out if d <= lim]
    return sorted(out, key=lambda od: (od[1], od[0].box.w * od[0].box.h))


def object_at(
    things: Sequence[Thing], p: Point, profile: Profile, label: str | None = None
) -> tuple[Thing | None, list[Thing]]:
    """The thing pointed at, or (None, candidates) when it is ambiguous or nothing is there.

    With a label ("that cat"), things near the point with that label are preferred."""
    cands = near(things, p, profile)
    if label:
        named = [(o, d) for o, d in cands if matches(o, label)]
        cands = named or cands
    if not cands:
        return None, []
    best, d1 = cands[0]
    if len(cands) >= 2:
        other, d2 = cands[1]
        nested = d1 == 0 and d2 == 0  # inside both: the smaller (sorted first) wins
        if not nested and d2 - d1 < profile.scene.ambiguous_mm:
            return None, [best, other]
    return best, [best]


# ---------------------------------------------------------------- places


def fits(c: Point, size: float, profile: Profile) -> bool:
    """A square of side `size` centred at c is inside the work area (0.5 mm slack)."""
    b = Box.from_center(c[0], c[1], size, size)
    d = profile.area
    return d.x - 0.5 <= b.x and b.x2 <= d.x2 + 0.5 and d.y - 0.5 <= b.y and b.y2 <= d.y2 + 0.5


def free(things: Sequence[Thing], c: Point, size: float, ignore: Iterable[str] = ()) -> bool:
    """A square of side `size` centred at c overlaps no thing (ignoring the ids given)."""
    skip = set(ignore)
    b = Box.from_center(c[0], c[1], size, size)
    for o in things:
        if o.id in skip:
            continue
        if iou(o.box, b) > 0 or overlap(o.box, b):
            return False
    return True


def _beside(t: Box, d: tuple[float, float], gap: float, half: tuple[float, float]) -> Point:
    """The centre of a box with half-sizes `half` placed `gap` beyond `t` along d (an axis)."""
    cx, cy = t.center
    hx, hy = half
    if d == (1.0, 0.0):
        return (t.x2 + gap + hx, cy)
    if d == (-1.0, 0.0):
        return (t.x - gap - hx, cy)
    if d == (0.0, 1.0):
        return (cx, t.y2 + gap + hy)
    return (cx, t.y - gap - hy)


def _ground(t: Box, c: Point, half: tuple[float, float], away: tuple[float, float]) -> Point:
    """`c` moved so the new box's base sits level with the target's base.

    On a page, things beside each other stand on the same ground: a tree put left of a house
    must have its base on the house's base, not its middle on the house's middle, or the two
    bases miss by half the difference in height (rahul, 8 Oct 2026: a 72.5 mm tree beside a
    67 mm house sat 2.75 mm low).

    The base is the edge away from `away` ("up the page" as the viewer sees it), so a page
    read from the far side grounds against its other edge."""
    if away[1] > 0:
        return (c[0], t.y + half[1])
    if away[1] < 0:
        return (c[0], t.y2 - half[1])
    if away[0] > 0:
        return (t.x + half[0], c[1])
    return (t.x2 - half[0], c[1])


def _neg(v: tuple[float, float]) -> tuple[float, float]:
    return (-v[0] + 0.0, -v[1] + 0.0)


def _axis(v: Any) -> tuple[float, float]:
    return (float(v[0]), float(v[1]))


def relation(
    rel: str,
    target: Thing,
    size: float,
    profile: Profile,
    things: Sequence[Thing] = (),
    target2: Thing | None = None,
    extent: tuple[float, float] | None = None,
) -> Spot:
    """Where a new thing of `size` (its larger side) goes for `rel` the target.

    With `extent` (the new thing's real width and height in the frame) it sits `gap_mm` from
    the target; without, it is treated as a square of side `size`. Raises `Unsupported` for
    a relation the view mode doesn't have."""
    t = target.box
    g = profile.scene.gap_mm
    half = (extent[0] / 2, extent[1] / 2) if extent else (size / 2, size / 2)
    right, away = _axis(profile.viewer.right), _axis(profile.viewer.away)
    view = profile.view
    dirs = {"right_of": right, "left_of": _neg(right), "behind": away, "in_front_of": _neg(away)}
    if view == "page":
        dirs |= {"above": away, "below": _neg(away)}
    side = view == "page" and rel in ("right_of", "left_of")
    if rel in dirs:
        c = _beside(t, dirs[rel], g, half)
        return Spot(_ground(t, c, half, away) if side else c)
    if rel == "above" and view == "surface":
        rel = "on"  # a block above another one stands on it
    if rel in ("above", "below"):
        raise Unsupported(f"'{rel}' has no meaning in the {view} view")
    if rel == "next_to":
        if view == "page":
            order = ["right_of", "left_of", "above", "below"]
        else:
            order = ["right_of", "left_of", "in_front_of", "behind"]
        options = {r: _beside(t, dirs[r], g, half) for r in order}
        if view == "page":  # the two sideways options stand on the target's ground
            options |= {r: _ground(t, options[r], half, away) for r in ("right_of", "left_of")}
        for r in order:
            c = options[r]
            if fits(c, size, profile) and free(things, c, size, [target.id]):
                return Spot(c)
        for r in order:
            if fits(options[r], size, profile):
                return Spot(options[r])
        return Spot(options["right_of"])
    if rel == "between":
        assert target2 is not None
        a, b = target.center, target2.center
        return Spot(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    if rel == "inside":
        return Spot(t.center)
    if rel == "on":
        if view != "page":  # stacked on top
            return Spot(t.center, target.height_mm)
        lift = (ON_LIFT * half[0], ON_LIFT * half[1])
        return Spot(_beside(t, away, 0.0, lift))
    raise ValueError(f"unknown relation {rel!r}")


def inside_size(target: Thing, profile: Profile) -> float:
    """'Inside': the new thing's larger side, a fraction of the target's smaller side."""
    return profile.scene.inside_fill * min(target.box.w, target.box.h)


def named(name: str, size: float, profile: Profile) -> Point | None:
    """A region of the work area in words: middle, top, bottom left corner, left side, ...

    'Left' and 'right' follow the viewer's right; 'top' and 'bottom' its away direction.
    None if the words name no region."""
    words = set(_norm(name).replace("-", " ").split())
    words |= {"top" for w in list(words) if w == "upper"}
    words |= {"bottom" for w in list(words) if w == "lower"}
    d = profile.area
    h = size / 2
    right, away = _axis(profile.viewer.right), _axis(profile.viewer.away)

    def end(v: tuple[float, float], positive: bool) -> tuple[int, float]:
        """(axis, coordinate) of the area's edge, inset by h, along +v or -v."""
        ax = 0 if v[0] != 0 else 1
        lo, hi = (d.x, d.x2) if ax == 0 else (d.y, d.y2)
        sign = v[ax] if positive else -v[ax]
        return ax, (hi - h) if sign > 0 else (lo + h)

    picks: dict[int, float] = {}
    for w, v, pos in (
        ("left", right, False),
        ("right", right, True),
        ("bottom", away, False),
        ("top", away, True),
    ):
        if w in words:
            ax, val = end(v, pos)
            picks.setdefault(ax, val)
    cx, cy = d.center
    if not picks:
        if words & {"middle", "centre", "center"}:
            return (cx, cy)
        return None
    return (picks.get(0, cx), picks.get(1, cy))


def chessboard_distance(free_cells: np.ndarray) -> np.ndarray:
    """For each free cell (non-zero), the chessboard distance to the nearest occupied cell,
    counting everything outside the grid as occupied; 0 for occupied cells.

    Each 3x3 erosion peels one layer, so a cell's distance is the number of erosions it
    survives plus one. This equals OpenCV's distanceTransform(DIST_C) on a zero-padded grid.
    """
    cur = np.asarray(free_cells) != 0
    dist = np.zeros(cur.shape, np.int32)
    while cur.any():
        dist += cur
        p = np.pad(cur, 1, constant_values=False)
        rows = p[:, :-2] & p[:, 1:-1] & p[:, 2:]
        cur = rows[:-2] & rows[1:-1] & rows[2:]
    return dist


def _down(v: float) -> int:
    """The largest integer not above v (rounding toward minus infinity)."""
    return -math.ceil(-v)


def empty_space(
    things: Sequence[Thing], size: float, profile: Profile, grid_mm: float | None = None
) -> tuple[Point, float] | None:
    """Centre and side of the largest empty square (every thing's box kept clear by the
    profile's margin). The side returned is at most `size`; None if nothing is free.
    Several equally good spots: the one nearest the middle of the area."""
    d = profile.area
    res = profile.scene.grid_mm if grid_mm is None else grid_mm
    w, h = int(d.w / res), int(d.h / res)
    if w <= 0 or h <= 0:
        return None
    free_cells = np.ones((h, w), np.uint8)
    m = profile.scene.empty_margin_mm
    for o in things:
        b = o.box
        x0 = _down((b.x - m - d.x) / res)
        x1 = int(math.ceil((b.x2 + m - d.x) / res))
        y0 = _down((b.y - m - d.y) / res)
        y1 = int(math.ceil((b.y2 + m - d.y) / res))
        free_cells[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0
    dist = chessboard_distance(free_cells)
    best = int(dist.max())
    if best <= 0:
        return None
    ys, xs = np.nonzero(dist >= best)
    cx0, cy0 = w / 2, h / 2
    k = int(np.argmin((xs - cx0) ** 2 + (ys - cy0) ** 2))
    side = (2 * best - 1) * res
    center = (d.x + (float(xs[k]) + 0.5) * res, d.y + (float(ys[k]) + 0.5) * res)
    return center, min(size, side)
