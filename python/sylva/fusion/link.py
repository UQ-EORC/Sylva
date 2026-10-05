# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Linking terrestrial and airborne tree maps."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ._common import _matrix, _transform_xyz


def _table(trees, what: str, columns: tuple) -> dict:
    """A dict of arrays from a table, a list of objects or an ``als.Trees``."""
    if isinstance(trees, Mapping):
        if "x" not in trees or "y" not in trees:
            raise ValueError(f"{what} needs 'x' and 'y' columns")
        n = len(np.asarray(trees["x"]))
        out = {k: np.asarray(v) for k, v in trees.items()}
        for k, v in out.items():
            if k != "crowns" and v.ndim == 1 and len(v) != n:
                raise ValueError(f"column {k!r} of {what} has {len(v)} values for {n} trees")
        return out
    if hasattr(trees, "table") and hasattr(trees, "crowns"):       # als.Trees
        t = {k: np.asarray(v) for k, v in trees.table().items()}
        t["crowns"] = list(trees.crowns)
        return t
    rows = list(trees)
    out = {}
    for c in columns:
        vals = [getattr(t, c, None) for t in rows]
        if c in ("x", "y") and any(v is None for v in vals):
            raise ValueError(f"{what} must be trees with x and y")
        if all(v is not None for v in vals):
            out[c] = np.asarray(vals)
    if "x" not in out:
        out["x"], out["y"] = np.zeros(0), np.zeros(0)
    return out


def _volumes(volumes, ids: np.ndarray, n: int) -> np.ndarray:
    if volumes is None:
        return np.full(n, np.nan)
    if hasattr(volumes, "volume") and hasattr(volumes, "models"):    # qsm.PlotQSMs
        return np.array([volumes.volume(int(i)) if int(i) in volumes.models else np.nan for i in ids])
    if isinstance(volumes, Mapping):
        return np.array([float(volumes.get(int(i), np.nan)) for i in ids])
    v = np.asarray(volumes, dtype=float).ravel()
    if len(v) != n:
        raise ValueError(f"{len(v)} volumes for {n} TLS trees")
    return v


@dataclass
class TreeLinks:
    """TLS trees linked to ALS trees, from :func:`link_trees`.

    Attributes
    ----------
    tls
        The TLS trees as a table: ``tree_id``, ``x``, ``y`` (in the ALS
        frame), ``dbh``, ``height`` (TLS), ``volume``.
    als
        The ALS trees as a table: ``id``, ``x``, ``y``, ``height`` and
        ``crowns`` when given.
    status
        Per TLS tree: ``matched`` (it is the ALS tree), ``suppressed``
        (under the crown of a taller ALS tree, which the ALS cannot see
        past), ``codominant`` (under an ALS crown whose tree is another
        stem of about the same height: the ALS merged two canopy trees) or
        ``unlinked`` (under no ALS crown and near no top).
    als_index
        Index into ``als`` of the tree each TLS tree is linked to or stands
        under; -1 for none.
    distance
        Stem to that tree's top (m).
    inside
        Whether the stem is inside that crown.
    als_height
        That ALS tree's height.
    height
        Combined height: for a matched tree the ALS height, unless the TLS
        is known to have seen the top and measured it taller; the TLS
        height otherwise.
    height_source
        ``"als"``, ``"tls"`` or ``"none"``.
    top_seen
        Whether the TLS saw the top: 1, 0 or NaN (unknown). Given, or for a
        matched tree inferred from the TLS height reaching the ALS height.
    flag
        ``top_seen``; ``als_height`` (TLS top not seen, ALS height used);
        ``top_not_seen`` (not seen and no ALS height: the height is a lower
        bound); ``top_unknown``.
    cost
        Assignment cost of each matched tree.
    als_tls
        Per ALS tree, the index of its matched TLS tree (-1 for none).
    als_stems
        Per ALS tree, the indices of every TLS tree linked to it or under it.
    settings
        The settings of the call.
    """

    tls: dict
    als: dict
    status: np.ndarray
    als_index: np.ndarray
    distance: np.ndarray
    inside: np.ndarray
    als_height: np.ndarray
    height: np.ndarray
    height_source: np.ndarray
    top_seen: np.ndarray
    flag: np.ndarray
    cost: np.ndarray
    als_tls: np.ndarray
    als_stems: list
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        c = self.counts()
        return (f"TreeLinks({len(self.status)} TLS trees: {c['matched']} matched, {c['suppressed']} "
                f"suppressed, {c['codominant']} codominant, {c['unlinked']} unlinked)")

    def counts(self) -> dict:
        """Numbers of TLS trees by status, of ALS trees with a matched stem
        and without, and of ALS crowns holding two stems or more."""
        out = {s: int(np.sum(self.status == s)) for s in ("matched", "suppressed", "codominant", "unlinked")}
        out["als_matched"] = int(np.sum(self.als_tls >= 0))
        out["als_unmatched"] = int(np.sum(self.als_tls < 0))
        out["one_to_many"] = int(sum(len(s) >= 2 for s in self.als_stems))
        return out

    def one_to_many(self) -> dict:
        """ALS trees with more than one TLS stem under them.

        Returns
        -------
        dict
            ALS tree id to the TLS tree ids under it, the matched one first.
        """
        out = {}
        for j, stems in enumerate(self.als_stems):
            if len(stems) >= 2:
                main = int(self.als_tls[j])
                order = [main] + [int(i) for i in stems if int(i) != main] if main >= 0 else [int(i) for i in stems]
                out[int(self.als["id"][j])] = [int(self.tls["tree_id"][i]) for i in order]
        return out

    def table(self) -> dict:
        """One row per TLS tree: its TLS measurements, its link and the
        combined height.

        Returns
        -------
        dict
            ``tree_id``, ``x``, ``y``, ``dbh``, ``height_tls``, ``volume``,
            ``status``, ``als_id`` (-1 for none), ``als_height``,
            ``distance``, ``inside``, ``height``, ``height_source``,
            ``top_seen``, ``flag``.
        """
        ids = np.asarray(self.als["id"])
        return {
            "tree_id": self.tls["tree_id"], "x": self.tls["x"], "y": self.tls["y"],
            "dbh": self.tls["dbh"], "height_tls": self.tls["height"], "volume": self.tls["volume"],
            "status": self.status,
            "als_id": np.where(self.als_index >= 0, ids[np.clip(self.als_index, 0, None)] if len(ids) else -1, -1),
            "als_height": self.als_height, "distance": self.distance, "inside": self.inside,
            "height": self.height, "height_source": self.height_source, "top_seen": self.top_seen,
            "flag": self.flag,
        }

    def als_table(self) -> dict:
        """One row per ALS tree: ``id``, ``x``, ``y``, ``height``,
        ``tls_id`` (its matched TLS tree, -1 for none), ``n_stems`` (TLS
        stems linked to it or under it) and ``stems`` (their ids)."""
        tid = np.asarray(self.tls["tree_id"])
        return {"id": self.als["id"], "x": self.als["x"], "y": self.als["y"], "height": self.als["height"],
                "tls_id": np.array([tid[i] if i >= 0 else -1 for i in self.als_tls], dtype=np.int64),
                "n_stems": np.array([len(s) for s in self.als_stems], dtype=np.int64),
                "stems": [tid[np.asarray(s, dtype=int)].tolist() for s in self.als_stems]}

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as CSV (empty cells for NaN).

        Parameters
        ----------
        path
            Output file.
        """
        t = self.table()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(t))
            for row in zip(*t.values(), strict=True):
                w.writerow(["" if isinstance(v, float) and not np.isfinite(v) else
                            (v.item() if isinstance(v, np.generic) else v) for v in row])


def link_trees(tls_trees, als_trees, transform=None, *, volumes=None, top_seen=None, sampling=None,
               min_above_observed: float = 0.5, max_distance: float = 3.0,
               crown_buffer: float = 0.5, height_weight: float = 2.0, dbh_weight: float = 2.0,
               max_cost: float = 2.0, top_tolerance: float = 1.0) -> TreeLinks:
    """Link TLS stems to ALS trees and combine their measurements.

    A TLS tree and an ALS tree can be one tree when the stem lies inside the
    ALS crown (or within ``crown_buffer`` of its outline) or within
    ``max_distance`` of its top. Among these pairs an optimal assignment
    (Kuhn 1955; Munkres 1957) minimises the total cost, a stem left without
    an ALS tree costing ``max_cost``. A pair costs::

        (d / D)² + height_weight * dh² + dbh_weight * (1 - dbh / dbh_max)²

    with ``d`` the stem-to-top distance, ``D`` the larger of
    ``max_distance`` and the crown's equivalent radius, ``dh = (h_als -
    h_tls) / h_als`` (0.3 for a stem without a TLS height) and ``dbh_max``
    the largest DBH among the stems that could be that ALS tree. The stem
    that is tallest by the TLS and thickest under a crown is therefore its
    tree, as the dominant tree of a crown usually is both. Each ALS tree
    gets at most one stem, and the others under its crown are reported with
    it as ``suppressed`` (shorter by more than ``top_tolerance`` and 10 %)
    or ``codominant`` (about as tall: the ALS saw two canopy trees as one).
    :meth:`TreeLinks.one_to_many` lists these crowns.

    The combined table keeps the TLS diameter and takes the height from the
    instrument that saw the top. Whether the TLS saw a tree's top comes from
    ``top_seen``, from ``sampling`` (:func:`sylva.voxels.tree_sampling`:
    seen when ``above_observed_fraction`` is at least
    ``min_above_observed``), or, for a matched tree, from its TLS height
    reaching the ALS height within ``top_tolerance``. A matched tree takes
    the ALS height unless the TLS is known (from ``top_seen`` or
    ``sampling``) to have seen the top and measured it taller: both are
    lower bounds, the ALS for missing the apex and the TLS for occlusion,
    but a TLS height can also be too tall where its segmentation gave the
    tree part of a neighbour's crown. Other trees keep the TLS height,
    flagged ``top_not_seen`` when that is a lower bound.

    Parameters
    ----------
    tls_trees
        :class:`sylva.trees.Tree` objects (after
        :func:`sylva.trees.tree_heights`), or a table (dict of arrays) with
        ``x``, ``y`` and optionally ``tree_id``, ``dbh``, ``height``,
        ``volume``, ``z``, in the TLS frame.
    als_trees
        :class:`sylva.als.Trees` (with crowns), or a table with ``x``,
        ``y``, ``height`` and optionally ``id`` and ``crowns`` (a list of
        ``(k, 2)`` outlines).
    transform
        :class:`Registration` or ``(4, 4)`` matrix taking the TLS into the
        ALS frame; the trees are taken to share a frame if None. Stem
        positions are moved at ``z`` (0 if not given).
    volumes
        Wood volume (m³) per TLS tree: a :class:`sylva.qsm.PlotQSMs`, a
        mapping of tree id to volume, or an array in the order of the trees.
    top_seen
        Per TLS tree, whether its top was seen (booleans, or 1, 0 and NaN).
    sampling
        Output of :func:`sylva.voxels.tree_sampling` (matched by
        ``tree_id``), in place of ``top_seen``.
    min_above_observed
        ``above_observed_fraction`` from which a top counts as seen.
    max_distance
        Farthest a stem can be from an ALS top (m) to be its tree outside
        the crown outline, and the scale of the distance cost.
    crown_buffer
        Distance (m) outside a crown outline that still counts as under it.
    height_weight
        Weight of the relative height difference in the cost.
    dbh_weight
        Weight of the DBH shortfall from the thickest candidate stem.
    max_cost
        Cost of leaving a stem without an ALS tree; no pair costing more is
        made.
    top_tolerance
        Height difference (m) within which a TLS tree reaches the ALS
        height.

    Returns
    -------
    TreeLinks

    Raises
    ------
    ValueError
        For a non-finite position, mismatched columns or a setting out of
        range.

    Examples
    --------
    >>> links = fusion.link_trees(stems, als_trees, transform=reg, volumes=qsms)  # doctest: +SKIP
    >>> links.one_to_many()                          # {als id: [tls ids]}         # doctest: +SKIP
    >>> links.to_csv("trees_fused.csv")                                           # doctest: +SKIP
    """
    t = _table(tls_trees, "tls_trees", ("tree_id", "x", "y", "dbh", "height", "volume", "z"))
    a = _table(als_trees, "als_trees", ("id", "x", "y", "height", "crowns"))
    n, m = len(np.asarray(t["x"])), len(np.asarray(a["x"]))
    col = lambda d, k, size: np.asarray(d[k], dtype=float) if k in d else np.full(size, np.nan)  # noqa: E731
    ids = np.asarray(t["tree_id"]) if "tree_id" in t else np.arange(1, n + 1)
    x, y = col(t, "x", n), col(t, "y", n)
    mat = _matrix(transform)
    if mat is not None and n:
        z = col(t, "z", n)
        xyz = _transform_xyz(mat, np.column_stack([x, y, np.nan_to_num(z)]))
        x, y = xyz[:, 0], xyz[:, 1]
    vol = col(t, "volume", n) if "volume" in t and volumes is None else _volumes(volumes, ids, n)
    if sampling is not None and top_seen is not None:
        raise ValueError("give top_seen or sampling, not both")
    if sampling is not None:
        frac = dict(zip(np.asarray(sampling["tree_id"]).tolist(),
                        np.asarray(sampling["above_observed_fraction"], dtype=float).tolist(), strict=True))
        f = np.array([frac.get(int(i), np.nan) for i in ids])
        seen = np.where(np.isfinite(f), (f >= min_above_observed).astype(float), np.nan)
    elif top_seen is not None:
        seen = np.asarray(top_seen, dtype=float).ravel()
        if len(seen) != n:
            raise ValueError(f"top_seen has {len(seen)} values for {n} TLS trees")
    else:
        seen = np.full(n, np.nan)
    tls_arr = np.column_stack([x, y, col(t, "dbh", n), col(t, "height", n), vol, seen]).reshape(-1, 6)
    als_ids = np.asarray(a["id"]) if "id" in a else np.arange(1, m + 1)
    als_arr = np.column_stack([col(a, "x", m), col(a, "y", m), col(a, "height", m)]).reshape(-1, 3)
    crowns = [np.ascontiguousarray(np.asarray(c, dtype=float).reshape(-1, 2)) for c in a["crowns"]] \
        if "crowns" in a else [np.zeros((0, 2)) for _ in range(m)]
    d = _core.fusion_link_trees(np.ascontiguousarray(tls_arr), np.ascontiguousarray(als_arr), crowns,
                                float(max_distance), float(crown_buffer), float(height_weight),
                                float(dbh_weight), float(max_cost), float(top_tolerance))
    tls_table = {"tree_id": ids, "x": x, "y": y, "dbh": col(t, "dbh", n), "height": col(t, "height", n),
                 "volume": vol}
    als_table = {"id": als_ids, "x": als_arr[:, 0], "y": als_arr[:, 1], "height": als_arr[:, 2]}
    if "crowns" in a:
        als_table["crowns"] = list(a["crowns"])
    return TreeLinks(
        tls=tls_table, als=als_table, status=np.array(d["status"], dtype=object).astype(str),
        als_index=d["als"], distance=d["distance"], inside=d["inside"], als_height=d["als_height"],
        height=d["height"], height_source=np.array(d["height_source"], dtype=object).astype(str),
        top_seen=d["top_seen"], flag=np.array(d["flag"], dtype=object).astype(str), cost=d["cost"],
        als_tls=d["als_tls"], als_stems=list(d["als_stems"]),
        settings={"max_distance": max_distance, "crown_buffer": crown_buffer,
                  "height_weight": height_weight, "dbh_weight": dbh_weight, "max_cost": max_cost,
                  "top_tolerance": top_tolerance,
                  "min_above_observed": min_above_observed})
