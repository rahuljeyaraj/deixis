"""Frame fits and transforms (F18): from one robot frame to another, in pure NumPy.

A robot side often needs a second frame: camera pixels to the work surface, or the camera's
ground plane to the robot's own map. `fit_transform` fits one from point pairs (at least 2 for a
similarity, 4 for a homography) and reports its residuals; `Transform.apply_where` moves a
`Where` (its point, box, path and covariance) into the other frame.

- **Similarity:** rotation, one scale and a shift (Umeyama, 1991). Optionally a mirror.
- **Homography:** a plane seen at an angle, by the normalised direct linear transform
  (Hartley and Zisserman, Multiple View Geometry, 4.4).

Fitting with outliers (RANSAC) is the robot side's job; give the core good pairs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import Field

from deixis.geometry import Cov2, Point
from deixis.where import Where, Wire

Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
TransformKind = Literal["similarity", "homography"]


class FitError(ValueError):
    """Too few or degenerate point pairs."""


def apply_matrix(m: Any, points: Any) -> np.ndarray:
    """Map (N, 2) points through a 3x3 matrix with the perspective divide."""
    h = np.asarray(m, dtype=np.float64).reshape(3, 3)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = pts @ h[:, :2].T + h[:, 2]
    w = q[:, 2:3]
    w = np.where(w == 0, np.inf, w)  # a point at infinity maps to 0, like OpenCV
    return q[:, :2] / w


class Transform(Wire):
    """A fitted map from one frame to another: a 3x3 matrix on (x, y, 1)."""

    kind: TransformKind
    matrix: Matrix3 = Field(description="3x3, row-major, acting on column vectors (x, y, 1).")
    from_frame: str | None = Field(default=None, description="The frame it maps from.")
    to_frame: str | None = Field(default=None, description="The frame it maps to.")
    rms_mm: float | None = Field(default=None, description="RMS residual of the fit (to_frame).")
    max_mm: float | None = Field(default=None, description="Largest residual of the fit.")
    pairs: int | None = Field(default=None, description="How many point pairs it was fitted on.")

    @property
    def m(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    def apply(self, points: Any) -> np.ndarray:
        """Map (N, 2) points (or one point) to the other frame; returns (N, 2)."""
        return apply_matrix(self.m, points)

    def apply_point(self, p: Sequence[float]) -> Point:
        x, y = self.apply(p)[0]
        return (float(x), float(y))

    def jacobian(self, p: Sequence[float]) -> np.ndarray:
        """The 2x2 derivative of the map at p (for moving a covariance)."""
        h = self.m
        x, y = float(p[0]), float(p[1])
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        u = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w
        v = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w
        return np.array(
            [
                [(h[0, 0] - u * h[2, 0]) / w, (h[0, 1] - u * h[2, 1]) / w],
                [(h[1, 0] - v * h[2, 0]) / w, (h[1, 1] - v * h[2, 1]) / w],
            ]
        )

    def inverse(self) -> Transform:
        inv = np.linalg.inv(self.m)
        inv = inv / inv[2, 2] if inv[2, 2] != 0 else inv
        return Transform(
            kind=self.kind,
            matrix=_tuple3(inv),
            from_frame=self.to_frame,
            to_frame=self.from_frame,
            pairs=self.pairs,
        )

    def apply_where(self, where: Where, to_frame: str | None = None) -> Where:
        """The same place in the other frame.

        The mean moves through the map; a box's 4 corners are mapped and their bounding box
        is returned; a path's points are mapped; the covariance becomes J C J^T, with J the
        map's derivative at the mean. Height (z) is kept as it is."""
        if self.from_frame is not None and where.frame != self.from_frame:
            raise ValueError(
                f"this transform maps from '{self.from_frame}', not from '{where.frame}'"
            )
        frame = to_frame or self.to_frame or where.frame
        assert where.mean_mm is not None
        mean = self.apply_point(where.mean_mm)
        jac = self.jacobian(where.mean_mm)
        cov = jac @ np.asarray(where.cov_mm2, dtype=np.float64) @ jac.T
        cov = (cov + cov.T) / 2
        box = None
        if where.box_mm is not None:
            x, y, w, h = where.box_mm
            corners = self.apply([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            box = (float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        path = None
        if where.path_mm is not None:
            path = [(float(a), float(b)) for a, b in self.apply(where.path_mm)]
        evidence = {**where.evidence, "from_frame": where.frame}
        return where.model_copy(
            update={
                "frame": frame,
                "mean_mm": mean,
                "box_mm": box,
                "path_mm": path,
                "cov_mm2": _cov(cov),
                "evidence": evidence,
            }
        )

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_json(cls, data: dict[str, Any] | str) -> Transform:
        return cls.model_validate(json.loads(data) if isinstance(data, str) else data)

    @classmethod
    def from_matrix(
        cls,
        m: Any,
        kind: TransformKind = "homography",
        from_frame: str | None = None,
        to_frame: str | None = None,
    ) -> Transform:
        return cls(
            kind=kind, matrix=_tuple3(np.asarray(m)), from_frame=from_frame, to_frame=to_frame
        )


def fit_transform(
    src: Any,
    dst: Any,
    kind: TransformKind = "similarity",
    from_frame: str | None = None,
    to_frame: str | None = None,
    allow_mirror: bool = False,
) -> Transform:
    """Fit a transform taking the `src` points onto the `dst` points (pairs in order).

    similarity: at least 2 pairs (rotation, uniform scale, shift; a mirror only with
    `allow_mirror`). homography: at least 4 pairs, no three on a line. The residuals
    (`rms_mm`, `max_mm`) are in the destination frame's units."""
    s = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    d = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(s) != len(d):
        raise FitError(f"{len(s)} source points but {len(d)} destination points")
    if kind == "similarity":
        m = _umeyama(s, d, allow_mirror)
    elif kind == "homography":
        m = _dlt(s, d)
    else:  # pragma: no cover - the Literal keeps this out
        raise FitError(f"unknown transform kind {kind!r}")
    res = np.linalg.norm(apply_matrix(m, s) - d, axis=1)
    return Transform(
        kind=kind,
        matrix=_tuple3(m),
        from_frame=from_frame,
        to_frame=to_frame,
        rms_mm=float(np.sqrt(np.mean(res**2))),
        max_mm=float(res.max()),
        pairs=len(s),
    )


def _umeyama(s: np.ndarray, d: np.ndarray, allow_mirror: bool) -> np.ndarray:
    if len(s) < 2:
        raise FitError("a similarity needs at least 2 point pairs")
    ms, md = s.mean(axis=0), d.mean(axis=0)
    sc, dc = s - ms, d - md
    var_s = float((sc**2).sum() / len(s))
    if var_s <= 1e-12:
        raise FitError("the source points are all in one place")
    cov = dc.T @ sc / len(s)
    u, sig, vt = np.linalg.svd(cov)
    e = np.eye(2)
    if not allow_mirror and np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[1, 1] = -1.0
    r = u @ e @ vt
    scale = float(np.trace(np.diag(sig) @ e) / var_s)
    t = md - scale * r @ ms
    m = np.eye(3)
    m[:2, :2] = scale * r
    m[:2, 2] = t
    return m


def _normaliser(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0)
    dist = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
    if dist <= 1e-12:
        raise FitError("the points are all in one place")
    k = np.sqrt(2.0) / dist
    return np.array([[k, 0.0, -k * c[0]], [0.0, k, -k * c[1]], [0.0, 0.0, 1.0]])


def _dlt(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    if len(s) < 4:
        raise FitError("a homography needs at least 4 point pairs")
    ts, td = _normaliser(s), _normaliser(d)
    sn, dn = apply_matrix(ts, s), apply_matrix(td, d)
    rows = []
    for (x, y), (u, v) in zip(sn, dn, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows)
    _, sv, vt = np.linalg.svd(a)
    if len(sv) >= 8 and sv[7] <= 1e-10 * sv[0]:
        raise FitError("degenerate point pairs (three or more on a line?)")
    hn = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ hn @ ts
    if abs(h[2, 2]) < 1e-15:
        raise FitError("degenerate homography")
    return h / h[2, 2]


def _tuple3(m: np.ndarray) -> Matrix3:
    m = np.asarray(m, dtype=np.float64).reshape(3, 3)
    return tuple(tuple(float(v) for v in row) for row in m)  # type: ignore[return-value]


def _cov(c: np.ndarray) -> Cov2:
    return ((float(c[0, 0]), float(c[0, 1])), (float(c[1, 0]), float(c[1, 1])))
