# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Two airborne surveys of one synthetic forest with known changes.

Part of :mod:`sylva.synthetic` (as :func:`sylva.synthetic.als_epochs`); it
composes :func:`~sylva.synthetic.stand`, :func:`~sylva.synthetic.crown_forest`
and :func:`~sylva.synthetic.als_flight`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pointcloud import PointCloud

__all__ = ["ALSEpochs", "als_epochs"]


@dataclass
class ALSEpochs:
    """Two airborne surveys of one synthetic forest; see :func:`als_epochs`.

    Attributes
    ----------
    scenes
        The scene of each survey (:func:`~sylva.synthetic.crown_forest`
        points and, with buildings, roof points of ``classification`` 6).
        Unchanged trees keep their points; a grown tree is rebuilt taller.
    flights
        The :class:`~sylva.synthetic.ALSFlight` of each survey, both in the
        true frame.
    trees
        One row per tree of the first survey (a table of equal-length
        arrays): ``tree_id``, stem ``x``, ``y``, true ``height_a`` and
        ``height_b`` (NaN once removed, from
        :func:`~sylva.synthetic.forest_trees`), ``top_x``, ``top_y`` and
        ``fate``: ``unchanged``, ``grown``, ``removed`` (felled singly) or
        ``gap`` (felled with its neighbours to open the gap).
    offset
        ``(dx, dy, dz)`` added to the second survey's points when its tiles
        are written: the misalignment an alignment should recover.
    gap
        ``(x, y, n)``: the gap was opened by felling the ``n`` trees nearest
        to ``(x, y)``.
    growth
        Height added to the grown trees (m).
    buildings
        ``(xmin, ymin, xmax, ymax)`` of each building.
    """

    scenes: list
    flights: list
    trees: dict
    offset: np.ndarray
    gap: tuple
    growth: float
    buildings: list = field(default_factory=list)

    def write_tiles(self, directory_a, directory_b, size: float = 50.0, epsg: int | None = None):
        """Write both surveys as tiles, the second displaced by ``offset``.

        Parameters
        ----------
        directory_a, directory_b
            Output directories.
        size
            Tile side (m).
        epsg
            EPSG code to record.

        Returns
        -------
        (Catalog, Catalog)
        """
        from . import als
        a = als.write_tiles(self.flights[0].points, directory_a, size, epsg=epsg, origin=(0.0, 0.0))
        pb = self.flights[1].points.copy()
        pb.xyz = np.ascontiguousarray(pb.xyz + self.offset)
        b = als.write_tiles(pb, directory_b, size, epsg=epsg, origin=(0.0, 0.0))
        return a, b


def _roof(x0, y0, x1, y1, base, rise, ridge_along_y: bool, spacing: float = 0.04) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-9, spacing)
    ys = np.arange(y0, y1 + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    if ridge_along_y:
        half = (x1 - x0) / 2
        z = base + rise * (1 - np.abs(gx - (x0 + half)) / half)
    else:
        half = (y1 - y0) / 2
        z = base + rise * (1 - np.abs(gy - (y0 + half)) / half)
    return np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])


def _balanced_scan_rate(f: dict) -> float:
    """Sweeps per second that make the spacing of pulses along the scan equal
    to that along the track, as airborne surveys are planned: ``speed /
    rate = swath * rate / pulse_rate``."""
    swath = 2 * f.get("altitude", 80.0) * np.tan(np.radians(f.get("scan_angle", 30.0)))
    return float(np.sqrt(f.get("pulse_rate", 50_000.0) * f.get("speed", 10.0) / swath))


def als_epochs(n_trees: int = 100, size: float = 100.0, removed: int = 4,
               gap=(70.0, 70.0, 5), growth: float = 1.0, offset=(0.3, -0.2, 0.15),
               buildings: bool = True, flight_a: dict | None = None,
               flight_b: dict | None = None, seed: int = 0) -> ALSEpochs:
    """Two airborne surveys of a synthetic forest with known changes.

    A stand of ``n_trees`` (:func:`~sylva.synthetic.stand`, 12 to 25 m,
    stems at least 4 m apart) in the west ``size`` m of a square of ``size +
    20`` m, with two gabled buildings east of it (one ridge running north,
    one east, so that their roofs have slopes of four aspects), is flown
    twice with :func:`~sylva.synthetic.als_flight`. Between the surveys

    - the ``gap[2]`` trees nearest to ``gap[:2]`` are felled, opening a gap;
    - ``removed`` other trees are felled singly;
    - the trees in the west half grow by ``growth`` m (their crowns scale
      with their height), the others are unchanged, point for point;
    - the second survey is flown by another sensor (by default 120 m above
      ground at 2,500 pulses/s with 40 m between lines, against 80 m at
      4,000 pulses/s and 30 m: about half the pulse density and a 50 %
      wider footprint);
    - its tiles are delivered displaced by ``offset``.

    Parameters
    ----------
    n_trees, size
        Trees and side (m) of the forest.
    removed
        Trees felled singly.
    gap
        ``(x, y, n)``: centre of the gap and trees felled to open it, or
        None for none.
    growth
        Height growth (m) of the trees in the west half.
    offset
        ``(dx, dy, dz)`` misalignment of the second survey.
    buildings
        Add the two buildings.
    flight_a, flight_b
        Settings of :func:`~sylva.synthetic.als_flight` overriding the
        defaults of each survey. Unless given, ``scan_rate`` is set so that
        pulses are as far apart along the scan as along the track, as
        surveys are planned (the default of ``als_flight``, 80 sweeps per
        second, leaves them 15 times farther apart across the track at these
        pulse rates).
    seed
        Random seed.

    Returns
    -------
    ALSEpochs
    """
    from . import synthetic
    rng = np.random.default_rng(seed)
    extent = size + 20.0
    trees = [t for t in synthetic.stand(n_trees, size=size, min_spacing=4.0, heights=(12.0, 25.0),
                                        seed=seed) if t[0] < size - 2.0]
    n = len(trees)
    xy = np.array([(t[0], t[1]) for t in trees]).reshape(-1, 2)
    fate = np.array(["unchanged"] * n, dtype="U9")
    fate[xy[:, 0] < size / 2] = "grown"
    if gap is not None:
        near = np.argsort(np.hypot(xy[:, 0] - gap[0], xy[:, 1] - gap[1]), kind="stable")
        fate[near[:int(gap[2])]] = "gap"
    free = np.flatnonzero(fate != "gap")
    if removed:
        fate[rng.choice(free, size=min(removed, len(free)), replace=False)] = "removed"
    scene_a = synthetic.crown_forest(trees, size=extent, ground_points=2000, margin=0.0, seed=seed)
    ids_a = np.asarray(scene_a.attrs["tree_id"])
    changed = np.isin(ids_a, np.flatnonzero(fate != "unchanged") + 1)
    scene_b = scene_a[~changed]
    grown = [i for i in range(n) if fate[i] == "grown"]
    if grown:
        taller = [(trees[i][0], trees[i][1], trees[i][2], trees[i][3] + growth) for i in grown]
        g = synthetic.crown_forest(taller, size=extent, ground_points=1, margin=0.0, seed=seed + 7)
        keep = np.asarray(g.attrs["classification"]) != 2
        g = g[keep]
        g.attrs["tree_id"] = np.asarray([grown[k - 1] + 1 for k in g.attrs["tree_id"]],
                                        dtype=np.int32)
        scene_b = PointCloud.concatenate([scene_b, g])
    boxes = []
    if buildings:
        roofs = []
        for (x0, y0, x1, y1, along_y) in ((size + 4, 15.0, size + 16, 30.0, True),
                                          (size + 4, 60.0, size + 16, 72.0, False)):
            base = float(synthetic.terrain_height((x0 + x1) / 2, (y0 + y1) / 2)) + 5.0
            roofs.append(_roof(x0, y0, x1, y1, base, 3.0, along_y))
            boxes.append((x0, y0, x1, y1))
        r = np.vstack(roofs)
        roof = PointCloud(r, {"classification": np.full(len(r), 6, dtype=np.uint8),
                              "tree_id": np.zeros(len(r), dtype=np.int32)})
        scene_a = PointCloud.concatenate([scene_a, roof])
        scene_b = PointCloud.concatenate([scene_b, roof])
    fa = dict(altitude=80.0, pulse_rate=4000.0, line_spacing=30.0, seed=seed)
    fb = dict(altitude=120.0, pulse_rate=2500.0, line_spacing=40.0, seed=seed + 1)
    fa.update(flight_a or {})
    fb.update(flight_b or {})
    for f in (fa, fb):
        f.setdefault("scan_rate", _balanced_scan_rate(f))
    bounds = (0.0, 0.0, extent, extent)
    flights = [synthetic.als_flight(scene_a, bounds=bounds, **fa),
               synthetic.als_flight(scene_b, bounds=bounds, **fb)]
    truth = [synthetic.forest_trees(s) for s in (scene_a, scene_b)]
    ta = truth[0]
    hb = np.full(n, np.nan)
    for tid, h in zip(truth[1]["tree_id"], truth[1]["height"], strict=True):
        if 1 <= tid <= n:
            hb[tid - 1] = h
    ha = np.full(n, np.nan)
    tx, ty = np.full(n, np.nan), np.full(n, np.nan)
    for k, tid in enumerate(ta["tree_id"]):
        if 1 <= tid <= n:
            ha[tid - 1] = ta["height"][k]
            tx[tid - 1], ty[tid - 1] = ta["top_x"][k], ta["top_y"][k]
    table = {"tree_id": np.arange(1, n + 1), "x": xy[:, 0], "y": xy[:, 1], "height_a": ha,
             "height_b": hb, "top_x": tx, "top_y": ty, "fate": fate}
    return ALSEpochs([scene_a, scene_b], flights, table, np.asarray(offset, dtype=float),
                     tuple(gap) if gap is not None else None, float(growth), boxes)
