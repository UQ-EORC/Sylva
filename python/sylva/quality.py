"""Point-cloud quality measured on tree stems.

Between 1 and 3 m a stem is the one surface in a forest scan whose shape is
known well enough to measure the scanner against. Each stem is cut into thin
slices and a circle fitted to every slice from all scans together; every
point's radial residual is then read per scan position::

    from sylva import quality

    q = quality.stem_noise(cloud, scan_id=quality.scan_ids_from_origins(origins))
    q.summary()      # sigma_total, sigma_within, sigma_local, registration, tails
    q.scans          # per scan: tx, ty (horizontal offset), sigma_within, sigma_local

What each spread contains:

``sigma_total``
    all points about the shared circle, as the cloud stands: range noise,
    bark, the stem's departure from a circle and misregistration together.
    A whole-stem circle cannot get below a few millimetres here even for a
    perfect scanner (about 7 mm on synthetic stems).
``sigma_within``
    one scan's points about their own median in the slice: range noise,
    bark and stem shape over the arc that scan saw; misregistration removed.
``sigma_local``
    what remains after removing a smooth curve along that arc (constant,
    first and second harmonics of the angle): range noise and bark only, the
    measure closest to the scanner's own noise.
``tx, ty``
    the horizontal offset of each scan, from ``sum (r - t . n)^2`` over its
    points on every stem (``n`` the outward normal), refined by moving the
    scans back and refitting; relative to the mean of all scans, since a
    common shift is invisible. Vertical misregistration does not show on
    vertical stems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["StemNoise", "stem_noise", "scan_ids_from_origins"]


def scan_ids_from_origins(origins, tolerance: float = 0.05) -> np.ndarray:
    """Scan position of each point from its sensor origin (one row per point,
    e.g. ``shots.origin[shots.shot_of_echo()]``): origins within
    ``tolerance`` metres are one position. Returns ``0 .. n-1``."""
    o = np.asarray(origins, dtype=float)
    key = np.round(o / tolerance).astype(np.int64)
    _, ids = np.unique(key, axis=0, return_inverse=True)
    return ids.ravel().astype(np.int64)


def _wmedian(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & (w > 0)
    if not ok.any():
        return float("nan")
    o = np.argsort(x[ok])
    c = np.cumsum(w[ok][o])
    return float(x[ok][o][np.searchsorted(c, c[-1] / 2)])


@dataclass
class StemNoise:
    slices: dict  #: per stem slice: stem, height, cx, cy, radius, n_points, sigma, sigma_first, arc, tail_fraction
    scan_slices: dict  #: per scan and slice: scan, slice, n_points, median_residual, sigma_within, sigma_local
    scans: dict  #: per scan: scan, n_points, n_slices, tx, ty, sigma_within, sigma_local
    residual: np.ndarray  #: per input point (NaN where not used), after the scans were moved back

    def summary(self) -> dict:
        """Plot-level figures (metres): see the module docstring."""
        sl, ss, sc = self.slices, self.scan_slices, self.scans
        n = len(sl["height"])
        out = {"n_stems": int(len(np.unique(sl["stem"]))) if n else 0, "n_slices": int(n),
               "n_scans": int(len(sc["scan"]))}
        if n == 0:
            return out
        out["sigma_total"] = _wmedian(sl["sigma_first"], sl["n_points"])
        out["sigma_corrected"] = _wmedian(sl["sigma"], sl["n_points"])
        out["sigma_within"] = _wmedian(ss["sigma_within"], ss["n_points"])
        out["sigma_local"] = _wmedian(ss["sigma_local"], ss["n_points"])
        out["tail_fraction"] = float(np.average(sl["tail_fraction"], weights=sl["n_points"]))
        if len(sc["scan"]) > 1:
            off = np.hypot(sc["tx"], sc["ty"])
            out["registration_rms"] = float(np.sqrt(np.average(off ** 2, weights=sc["n_points"])))
            out["registration_max"] = float(off.max())
            out["worst_scan"] = int(sc["scan"][np.argmax(off)])
        return out


def stem_noise(cloud: PointCloud, scan_id=None, stems=None, height_attr: str = "height",
               iterations: int = 3, **params) -> StemNoise:
    """Stem-based noise and registration of ``cloud``.

    ``scan_id``: per-point scan position (array, the name of an attribute, or
    ``None`` for a single scan; see :func:`scan_ids_from_origins`).
    ``stems``: stem positions as ``(n, 2)`` xy, a list of
    :class:`sylva.trees.Tree`, or ``None`` to detect them. Heights come from
    ``height_attr`` if the cloud has it, else from the lowest point.
    ``params``: ``height_min=1``, ``height_max=3``, ``step=0.25``,
    ``thickness=0.1`` (slices, m above ground), ``min_radius=0.05``,
    ``max_radius=1``, ``min_arc=270`` (deg a slice's circle must see),
    ``min_inlier_fraction=0.5``, ``cut_min=0.05`` and ``cut_fraction=0.3``
    (points further than ``max(cut_min, cut_fraction * r)`` from the circle
    are not stem), ``min_scan_points=30``. ``iterations`` of offset refinement.
    """
    xyz = np.ascontiguousarray(cloud.xyz, dtype=float)
    if height_attr in cloud.attrs:
        h = np.ascontiguousarray(cloud.attrs[height_attr], dtype=float)
    else:
        h = xyz[:, 2] - xyz[:, 2].min()
    if isinstance(scan_id, str):
        scan_id = cloud.attrs[scan_id]
    ids = None if scan_id is None else np.ascontiguousarray(scan_id, dtype=np.int64)
    if stems is None:
        from . import trees

        found = trees.detect_stems(cloud if height_attr in cloud.attrs else cloud.with_attrs(**{height_attr: h}),
                                   height_attr=height_attr)
        stems = [(t.x, t.y) for t in found]
    elif len(stems) and hasattr(stems[0], "x"):
        stems = [(t.x, t.y) for t in stems]
    st = np.ascontiguousarray(np.asarray(stems, dtype=float).reshape(-1, 2))
    d = _core.stem_noise(xyz, h, st, ids, iterations=int(iterations), **params)
    return StemNoise(d["slices"], d["scan_slices"], d["scans"], d["residual"])
