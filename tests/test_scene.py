"""Things, 'that one', relations in each view mode, named regions, empty space (C7)."""

import itertools
import random

import numpy as np
import pytest

from deixis.profile import Profile
from deixis.scene import (
    Thing,
    Unsupported,
    chessboard_distance,
    empty_space,
    find,
    fits,
    free,
    labels,
    last,
    named,
    near,
    nearest,
    object_at,
    relation,
)


def thing(id, box, label=None, **kw):
    return Thing(id=id, label=label or id, box_mm=box, **kw)


def prof(view="page", right=(1, 0), away=(0, 1), area=(0, 0, 200, 200), **scene):
    return Profile.model_validate(
        {
            "name": "test", "frame": "f", "area_mm": area, "view": view,
            "viewer": {"right": right, "away": away}, "scene": scene,
        }
    )  # fmt: skip


HOUSE = thing("house", (40, 40, 40, 30), t=1)
TREE = thing("tree", (120, 40, 20, 50), t=2)


def test_that_one_is_the_nearest_selectable_thing():
    p = prof()
    label = thing("note", (40, 30, 40, 8), kind="label", selectable=False, t=3)
    assert object_at([HOUSE, TREE, label], (60, 35), p) == (HOUSE, [HOUSE])
    assert object_at([HOUSE, TREE], (100, 55), p)[0] is None  # 20 mm from both: ask
    assert object_at([HOUSE, TREE], (100, 55), p)[1] == [TREE, HOUSE]  # tie: smaller first
    assert object_at([HOUSE, TREE], (10, 190), p) == (None, [])
    assert [t.id for t, _ in near([HOUSE, TREE], (100, 55), p)] == ["tree", "house"]


def test_that_cat_prefers_the_named_thing():
    p = prof()
    cat = thing("cat", (100, 40, 15, 15), label="cat")
    dog = thing("dog", (118, 40, 15, 15), label="dog")
    best, cands = object_at([cat, dog], (116.5, 47), p)
    assert best is None and len(cands) == 2  # equally near: ask
    assert object_at([cat, dog], (116.5, 47), p, label="that cat")[0] is cat


def test_find_by_label_and_attributes():
    red = thing("red", (0, 0, 50, 50), label="red block", attrs={"color": "red"}, t=1)
    blue = thing("blue", (60, 0, 50, 50), label="blue block", attrs={"color": "blue"}, t=2)
    blue2 = thing("blue2", (120, 0, 50, 50), label="block", attrs={"color": "blue"}, t=3)
    assert find([red, blue, blue2], "the blue one") == [blue2, blue]  # most recent first
    assert find([red, blue, blue2], "red block") == [red]
    assert find([red, blue, blue2], "blocks") == [blue2]  # exact label first
    assert find([red, blue, blue2], "green one") == []
    assert find([red], "one") == []


def test_last_and_labels():
    arrow = thing("arrow", (0, 0, 5, 5), selectable=False, t=9)
    assert last([HOUSE, TREE, arrow]) is TREE
    assert last([HOUSE, TREE], last_id="house") is HOUSE
    assert last([arrow]) is None
    assert labels([TREE, HOUSE, arrow]) == ["house", "tree"]


def test_fits_and_free():
    p = prof()
    assert fits((100, 100), 50, p) and not fits((10, 10), 50, p)
    assert free([HOUSE], (150, 150), 20) and not free([HOUSE], (60, 55), 10)
    assert free([HOUSE], (60, 55), 10, ignore=["house"])


@pytest.mark.parametrize(
    "rel,view,expect",
    [
        ("left_of", "page", (40 - 10 - 10, 40 + 10)),  # on the house's ground, not its middle
        ("right_of", "surface", (80 + 10 + 10, 55)),
        ("above", "page", (60, 70 + 10 + 10)),
        ("below", "page", (60, 40 - 10 - 10)),
        ("behind", "page", (60, 90)),
        ("behind", "surface", (60, 90)),
        ("in_front_of", "ground", (60, 20)),
        ("inside", "ground", (60, 55)),
        ("on", "page", (60, 70 + 0.7 * 10)),
    ],
)
def test_relations_by_view(rel, view, expect):
    spot = relation(rel, HOUSE, 20, prof(view, gap_mm=10))
    assert spot.center == pytest.approx(expect) and spot.z_mm is None


def test_on_stacks_on_a_surface():
    green = thing("green", (300, 100, 50, 50), height_mm=50)
    p = prof("surface", area=(0, 0, 600, 400))
    assert relation("on", green, 50, p) == ((325, 125), 50)
    assert relation("above", green, 50, p) == ((325, 125), 50)  # above = on, stacked
    with pytest.raises(Unsupported):
        relation("below", green, 50, p)
    with pytest.raises(Unsupported):
        relation("above", green, 50, prof("ground", area=(0, 0, 600, 400)))
    assert relation("on", green, 50, prof("ground", area=(0, 0, 600, 400))).z_mm == 50


def test_next_to_tries_the_sides_in_order():
    p = prof("page", gap_mm=10)
    assert relation("next_to", HOUSE, 20, p).center == (100, 50)  # right, on its ground
    blocker = thing("x", (85, 40, 30, 30))
    assert relation("next_to", HOUSE, 20, p, [HOUSE, blocker]).center == (20, 50)  # left
    q = prof("surface", gap_mm=10)
    wall = thing("w", (0, 40, 40, 30))
    # surface: right, left, then in front (toward the viewer), then behind
    assert relation("next_to", HOUSE, 20, q, [HOUSE, blocker, wall]).center == (60, 20)


def test_between_and_extent():
    p = prof()
    assert relation("between", HOUSE, 10, p, target2=TREE).center == (95, 60)
    assert relation("right_of", HOUSE, 30, p, extent=(10, 30)).center == (95, 55)


def test_viewer_turned_round():
    """The person stands at the far side: their right is -x and away is -y."""
    p = prof(right=(-1, 0), away=(0, -1), gap_mm=10)
    assert relation("left_of", HOUSE, 20, p).center == (100, 60)  # ground is the far edge now
    assert relation("above", HOUSE, 20, p).center == (60, 20)
    assert named("top left corner", 20, p) == (190, 10)


def test_named_regions():
    p = prof()
    assert named("top left corner", 20, p) == (10, 190)
    assert named("the lower right", 40, p) == (180, 20)
    assert named("left side", 20, p) == (10, 100)
    assert named("middle", 20, p) == (100, 100)
    assert named("on the roof", 20, p) is None
    rotated = prof(right=(0, 1), away=(-1, 0))  # right along +y, away along -x
    assert named("top right", 20, rotated) == (10, 190)


def brute_empty(free_cells):
    """For each free cell, the largest k such that the (2k-1)-square around it is free."""
    h, w = free_cells.shape
    out = np.zeros((h, w), np.int32)
    for y, x in itertools.product(range(h), range(w)):
        k = 0
        while True:
            r = k
            if y - r < 0 or x - r < 0 or y + r >= h or x + r >= w:
                break
            if not free_cells[y - r : y + r + 1, x - r : x + r + 1].all():
                break
            k += 1
        out[y, x] = k
    return out


def test_chessboard_distance_equals_brute_force():
    rng = random.Random(5)
    for _ in range(60):
        h, w = rng.randint(1, 15), rng.randint(1, 15)
        cells = np.array([[rng.random() > 0.25 for _ in range(w)] for _ in range(h)])
        assert np.array_equal(chessboard_distance(cells), brute_empty(cells))


def test_empty_space():
    p = prof(empty_margin_mm=0, grid_mm=1)
    assert empty_space([], 50, p) == ((100.5, 100.5), 50)  # side capped at the size asked
    left_half = thing("wall", (0, 0, 100, 200))
    (c, side) = empty_space([left_half], 500, p)
    assert side == 99 and 100 < c[0] < 200
    assert empty_space([thing("all", (-5, -5, 210, 210))], 20, p) is None
    coarse = prof(area=(0, 0, 3000, 2500), grid_mm=25, empty_margin_mm=100)
    sofa = thing("sofa", (0, 0, 3000, 1000))
    (c, side) = empty_space([sofa], 1000, coarse)
    assert side == 1000 and c[1] > 1100


def test_nearest():
    assert nearest((0, 0), []) is None
    assert nearest((0, 0), [(3, 4), (1, 1), (1, 1)]) == (1, pytest.approx(2**0.5))


def test_beside_on_a_page_stands_on_the_same_ground():
    """A tall thing put beside a short one keeps its base on the short one's base.

    Live on 8 Oct 2026 a 72.5 mm tree left of a 67 mm house was centred on the house, so the
    two bases missed by 2.75 mm and the tree floated below the ground line.
    """
    p = prof("page", gap_mm=10)
    tall = relation("right_of", HOUSE, 50, p, extent=(30, 50))
    assert tall.center == (80 + 10 + 15, 40 + 25)  # base on the house's base (y 40)
    short = relation("right_of", HOUSE, 10, p, extent=(10, 10))
    assert short.center == (80 + 10 + 5, 40 + 5)
    # Both bases are level with the house's, whatever their height.
    assert tall.center[1] - 25 == short.center[1] - 5 == HOUSE.box.y


def test_beside_on_a_surface_still_keeps_the_depth():
    """On a floor or table, "right of" must not drag a thing forward: depth is not height."""
    p = prof("surface", gap_mm=10)
    spot = relation("right_of", HOUSE, 50, p, extent=(30, 50))
    assert spot.center == (80 + 10 + 15, 55)  # the target's own depth, unchanged
