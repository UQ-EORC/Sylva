# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
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
    points on every stem (``n`` the outward normal; Huber 1964 weights),
    refined by moving the scans back and refitting; relative to the mean of
    all scans, since a common shift is invisible. Vertical misregistration
    does not show on vertical stems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = ["StemNoise", "stem_noise", "scan_ids_from_origins"]


def scan_ids_from_origins(origins, tolerance: float = 0.05) -> np.ndarray:
    """Scan position of each point, from its sensor origin.

    Parameters
    ----------
    origins
        ``(N, 3)`` sensor position of each point, e.g.
        ``shots.origin[shots.shot_of_echo()]`` or a ray cloud's end point
        plus ``nx, ny, nz``.
    tolerance
        Origins on the same ``tolerance`` (m) grid cell are one position.

    Returns
    -------
    numpy.ndarray
        int64 id per point, ``0 .. n-1`` in sorted order of the origins.
    """
    o = np.asarray(origins, dtype=float)
    o = np.ascontiguousarray(o.reshape(o.shape[0], int(np.prod(o.shape[1:]))))
    return _core.scan_ids_from_origins(o, float(tolerance))


def _wmedian(x, w):
    x = np.ascontiguousarray(np.ravel(np.asarray(x, float)))
    w = np.ascontiguousarray(np.ravel(np.asarray(w, float)))
    return _core.weighted_median(x, w)


@dataclass
class StemNoise:
    """Result of :func:`stem_noise`; tables are dicts of equal-length arrays.

    Attributes
    ----------
    slices
        Per stem slice: ``stem``, ``height``, ``cx``, ``cy``, ``radius``,
        ``n_points``, ``sigma`` (after moving scans back), ``sigma_first``
        (as the cloud stands), ``arc`` (degrees seen) and ``tail_fraction``.
    scan_slices
        Per scan and slice: ``scan``, ``slice``, ``n_points``,
        ``median_residual``, ``sigma_within``, ``sigma_local``.
    scans
        Per scan: ``scan``, ``n_points``, ``n_slices``, ``tx``, ``ty``
        (horizontal offset, m), ``sigma_within``, ``sigma_local``.
    residual
        Radial residual of each input point (m); NaN where not used.
    """

    slices: dict  #: per stem slice: stem, height, cx, cy, radius, n_points, sigma, sigma_first, arc, tail_fraction
    scan_slices: dict  #: per scan and slice: scan, slice, n_points, median_residual, sigma_within, sigma_local
    scans: dict  #: per scan: scan, n_points, n_slices, tx, ty, sigma_within, sigma_local
    residual: np.ndarray  #: per input point (NaN where not used), after the scans were moved back

    def summary(self, min_scan_slices: int = 1) -> dict:
        """Plot-level quality figures, in metres.

        Parameters
        ----------
        min_scan_slices
            Scans measured in fewer stem slices than this are left out of the
            registration figures. A scan that saw only a few stem points (an
            outer position looking away from the plot) gets an offset fitted
            to almost nothing, which would otherwise set ``registration_max``.

        Returns
        -------
        dict
            ``n_stems``, ``n_slices``, ``n_scans``; point-weighted medians
            ``sigma_total``, ``sigma_corrected``, ``sigma_within``,
            ``sigma_local``; ``tail_fraction``; and with several scans
            ``registration_rms``, ``registration_max`` and ``worst_scan``
            over the ``n_scans_registered`` scans with enough slices.
            Only the counts are present if no slice qualified.

        Notes
        -----
        On the TERN CUP 2022 plot ``sigma_local`` was 6.7 mm and
        ``registration_rms`` 4.7 mm; two halves of the stems agreed to
        1.1 mm. Compare scans and dates with the same settings, not with
        a fixed threshold.
        """
        sl, ss, sc = self.slices, self.scan_slices, self.scans
        n = len(sl["height"])

        def cols(d, keys, dtype=float):
            return {k: np.ascontiguousarray(d[k] if n else d.get(k, ()), dtype=dtype) for k in keys}

        slices = {**cols(sl, ["stem"], np.int64), **cols(sl, ["n_points", "sigma", "sigma_first", "tail_fraction"])}
        scan_slices = cols(ss, ["n_points", "sigma_within", "sigma_local"])
        scans = {"scan": np.ascontiguousarray(sc["scan"], dtype=np.int64),
                 **cols(sc, ["n_points", "n_slices", "tx", "ty"])}
        return _core.stem_noise_summary(n, slices, scan_slices, scans, float(min_scan_slices))


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

    Parameters
    ----------
    cloud
        Registered multi-scan cloud (or one scan) covering 1-3 m of the stems.
    scan_id
        Scan position per point, as above.
    stems
        Stem positions, as above.
    height_attr
        Attribute holding height above ground.
    iterations
        Rounds of moving the scans back and refitting.
    **params
        The slice and fit settings above.

    Returns
    -------
    StemNoise
        Call :meth:`StemNoise.summary` for plot-level numbers.

    Notes
    -----
    Needs stems that are roughly round and seen from most sides: at least
    ``min_arc`` degrees of each slice, so single scans of isolated trees
    usually qualify only with a lower ``min_arc``.
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
