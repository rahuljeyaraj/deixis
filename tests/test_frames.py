"""Frame fits and transforms (C8): the angled-view spike, similarity with noise, Where."""

import math

import numpy as np
import pytest

from deixis.frames import FitError, Transform, apply_matrix, fit_transform
from deixis.where import Where

# --- the 4 Oct spike: a 600 x 400 mm table seen at an angle by a pinhole camera -----------

W, H_IMG, FOV_Y = 1280, 720, 50.0
TABLE = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=float)


def camera(pitch_deg, eye_h=450.0, back=250.0):
    c = np.array([300.0, -back, eye_h])
    p = np.radians(pitch_deg)
    fwd = np.array([0, np.cos(p), -np.sin(p)])
    right = np.array([1.0, 0, 0])
    up = np.cross(right, fwd)
    f = (H_IMG / 2) / np.tan(np.radians(FOV_Y / 2))
    return c, fwd, right, up, f


def project(cam, pts_mm):
    c, fwd, right, up, f = cam
    p = np.c_[pts_mm, np.zeros(len(pts_mm))] - c
    z = p @ fwd
    return np.c_[W / 2 + f * (p @ right) / z, H_IMG / 2 - f * (p @ up) / z]


@pytest.mark.parametrize("pitch", [30, 40, 55, 90])
def test_four_screen_corners_map_pixels_to_table_mm_exactly(pitch):
    cam = camera(pitch, eye_h=450 if pitch < 90 else 900, back=250 if pitch < 90 else -200)
    t = fit_transform(project(cam, TABLE), TABLE, "homography", "screen", "table")
    gx, gy = np.meshgrid(np.linspace(10, 590, 30), np.linspace(10, 390, 20))
    grid = np.c_[gx.ravel(), gy.ravel()]
    assert np.abs(t.apply(project(cam, grid)) - grid).max() < 1e-6
    assert t.rms_mm < 1e-6 and t.pairs == 4


def test_mm_per_pixel_far_edge_matches_the_spike():
    """About 1.7 mm per pixel along the depth at the far edge (40 degree pitch)."""
    cam = camera(40)
    t = fit_transform(project(cam, TABLE), TABLE, "homography")
    q = project(cam, np.array([[300.0, 380.0]]))[0]
    depth = np.linalg.norm(t.apply(q + [0, 3]) - [300, 380]) / 3
    assert 1.3 < depth < 2.1


# --- similarity -----------------------------------------------------------------------------


def _similar(pts, angle_deg, scale, shift, mirror=False):
    a = math.radians(angle_deg)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = np.asarray(pts, float) * ([1, -1] if mirror else [1, 1])
    return scale * p @ r.T + shift


def test_similarity_recovered_with_5_mm_noise():
    rng = np.random.default_rng(1)
    floor = rng.uniform(0, 3000, size=(40, 2))
    noisy_map = _similar(floor, 30, 1.2, (500, -300)) + rng.normal(0, 5, size=(40, 2))
    t = fit_transform(floor, noisy_map, "similarity", "floor", "map")
    m = t.m
    scale = math.hypot(m[0, 0], m[1, 0])
    angle = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    assert scale == pytest.approx(1.2, abs=0.005) and angle == pytest.approx(30, abs=0.3)
    assert 4 < t.rms_mm < 10 and t.max_mm < 25
    truth = _similar([[1500, 1000]], 30, 1.2, (500, -300))[0]
    assert np.linalg.norm(t.apply_point((1500, 1000)) - truth) < 5


def test_similarity_from_two_points_and_mirror():
    t = fit_transform([(0, 0), (100, 0)], [(10, 10), (10, 110)])  # 90 degrees, shift
    assert t.apply_point((50, 0)) == pytest.approx((10, 60))
    pts = [(0, 0), (100, 0), (0, 50), (70, 80)]
    mirrored = _similar(pts, 10, 1.0, (5, 5), mirror=True)
    assert fit_transform(pts, mirrored).rms_mm > 10  # no mirror by default
    assert fit_transform(pts, mirrored, allow_mirror=True).rms_mm < 1e-9


def test_apply_where_moves_box_path_and_covariance():
    rot = fit_transform([(0, 0), (1, 0)], [(0, 0), (0, 2)], "similarity", "floor", "map")
    box = Where(
        frame="floor", kind="box", box_mm=(10, 0, 20, 10), cov_mm2=((4, 0), (0, 1)),
        source="gesture", evidence={"rule": "nearest"},
    )  # fmt: skip
    got = rot.apply_where(box)
    assert got.frame == "map" and got.kind == "box"
    assert got.box_mm == pytest.approx((-20, 20, 20, 40))  # rotated 90 degrees, scaled 2
    assert got.mean_mm == pytest.approx((-10, 40))
    assert np.allclose(got.cov_mm2, [[4, 0], [0, 16]])  # J C J^T with J = 2R
    assert got.evidence == {"rule": "nearest", "from_frame": "floor"}
    path = Where(frame="floor", kind="path", path_mm=[(0, 0), (5, 0)], source="geometry")
    assert rot.apply_where(path).path_mm == [pytest.approx((0, 0)), pytest.approx((0, 10))]
    with pytest.raises(ValueError, match="maps from 'floor'"):
        rot.apply_where(box.model_copy(update={"frame": "paper"}))


def test_inverse_json_and_errors():
    cam = camera(40)
    t = fit_transform(project(cam, TABLE), TABLE, "homography", "screen", "table")
    back = t.inverse()
    assert back.from_frame == "table" and back.to_frame == "screen"
    px = project(cam, np.array([[123.0, 321.0]]))
    assert np.allclose(back.apply(t.apply(px)), px)
    again = Transform.from_json(t.model_dump_json())
    assert again == t and Transform.from_json(t.to_json()) == t
    with pytest.raises(FitError, match="at least 4"):
        fit_transform(TABLE[:3], TABLE[:3], "homography")
    with pytest.raises(FitError, match="line"):
        fit_transform([(0, 0), (1, 1), (2, 2), (3, 3)], TABLE, "homography")
    with pytest.raises(FitError, match="one place"):
        fit_transform([(1, 1), (1, 1)], [(0, 0), (1, 1)])
    with pytest.raises(FitError, match="destination"):
        fit_transform([(1, 1), (2, 2)], [(0, 0)])


def test_apply_matrix_point_at_infinity_is_zero():
    m = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    assert apply_matrix(m, [(0, 5)]).tolist() == [[0.0, 0.0]]
