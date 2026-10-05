# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Trees of a tiled plot: segmentation, per-tree access and per-tree models.

The functions here are those of :mod:`sylva.als.tiles` for trees (they are
imported there and documented under that module):

1. :func:`segment_trees` segments the trees of height-normalised tiles tile
   by tile, with the labels :func:`sylva.trees.segment_trees` gives the whole
   plot, and writes them to the tiles.
2. :func:`split_trees` writes each tree's points to a :class:`TreeStore`, so
   that one tree is read without the plot (:func:`read_tree`).
3. :func:`classify_leaf_wood`, :func:`build_qsms` and :func:`crown_metrics`
   work tree by tree from the store, a few trees at a time, with the results
   of their whole-plot counterparts; leaf and wood go back to the tiles.
4. :func:`run_plot` runs the whole workflow from registered scans, saving
   every stage so that a second run resumes where the first stopped.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from .. import filters, leaves, qsm, trees
from ..util import limits, progress
from .catalogue import Catalog, _as_catalog, _format, _workers, _written, catalog
from ..pointcloud import PointCloud
from ..raster import Raster
from ..trees import Tree

__all__ = ["segment_trees", "TreeStore", "split_trees", "read_tree", "classify_leaf_wood",
           "build_qsms", "crown_metrics", "PlotRun", "run_plot"]

#: Bytes held per point of a tree while it is processed (the points, their
#: attributes and the working copies of a QSM fit or a leaf / wood graph).
BYTES_PER_TREE_POINT = 1024


def _record(info) -> None:
    from . import tiles
    tiles._record(info)


def _options(value, name: str) -> dict | None:
    """``True`` as the defaults, ``False`` or None as off, or a dict of keywords."""
    if value is True:
        return {}
    if value is False or value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{name} must be True, False or a dict of keyword arguments, got {value!r}")


def _origin(v) -> tuple[float, float, float]:
    o = tuple(float(x) for x in v)
    if len(o) != 3 or not all(np.isfinite(o)):
        raise ValueError(f"voxel_origin must be a finite (x, y, z), got {v!r}")
    return o


# ------------------------------------------------------------------ segmentation


def segment_trees(catalog: Catalog, stems: list[Tree], out: str | Path, height_attr: str = "height",
                  buffer: float = 20.0, max_buffer: float = 60.0, merge=True,
                  percentile: float = 100.0, prune=True, attribute: str = "tree_id",
                  voxel_origin=(0.0, 0.0, 0.0), edge_margin: float = 2.0,
                  workers: int | None = None, format: str | None = None,
                  **params) -> tuple[list[Tree], Catalog]:
    """Segment the trees of height-normalised tiles, as on the whole plot.

    The whole-plot sequence is :func:`sylva.trees.merge_branches`,
    :func:`sylva.trees.segment_trees`, :func:`sylva.trees.tree_heights` and
    :func:`sylva.trees.prune_trees`. Here each tree is segmented by the tile
    whose core holds its stem, from the graph nodes of that tile and of a
    ``buffer`` around it, and every point gets the label of its nearest node;
    the tiles are written to ``out`` with each point's tree id in
    ``attribute`` (-1 for none). Tree ids are those the whole-plot sequence
    gives, unique over the plot.

    The graph is built on the points above ``cut_above_ground`` thinned to
    one per voxel of a grid with a corner at ``voxel_origin``, found tile by
    tile exactly as in the whole cloud. With a buffer wider than the largest
    crown (every tree that competes with a tile's trees for points then has
    its stem, seeds and crown within the buffer), the labels, heights and
    trees are those of the whole-plot sequence with the same
    ``voxel_origin``, up to ties in the nearest-neighbour searches. A tile
    whose trees reach the edge of its buffer (a node within ``edge_margin``
    of it, where the plot goes on) is read again with a buffer twice as wide,
    up to ``max_buffer``; trees still at the edge are named in a warning and
    flagged ``crown_at_edge`` in their ``extra``.

    Parameters
    ----------
    catalog
        Height-normalised tiles (:func:`sylva.als.tiles.normalize`).
    stems
        Stems from :func:`sylva.als.tiles.detect_stems` (or any trees with
        unique ids, positions and DBH).
    out
        Directory for the segmented tiles.
    height_attr
        Attribute holding heights; z is used if the tiles do not have it.
    buffer
        Band of neighbouring nodes read around each tile (m); wider than the
        largest crown.
    max_buffer
        Widest band a tile is read again with (m).
    merge
        Merge branches into their stems first (:func:`sylva.trees.merge_branches`):
        True with its defaults, False to skip, or a dict of its keywords
        (``voxel_size``, ``k``, ``search_radius``, ...).
    percentile
        Height percentile of :func:`sylva.trees.tree_heights`.
    prune
        Prune afterwards (:func:`sylva.trees.prune_trees`): True with its
        defaults, False to skip, or a dict of its keywords (``min_height``,
        ``min_slenderness``, ...).
    attribute
        Name of the label attribute written to the tiles (int32).
    voxel_origin
        Corner of the voxel grid of the graphs.
    edge_margin
        How close to the edge of the buffer (m) a tree's nodes may come
        before its tile is read again.
    workers
        Tiles processed at once; the number of CPUs if None, fewer if memory
        is short. The result does not depend on it.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.
    **params
        Keywords of :func:`sylva.trees.segment_trees` (``voxel_size``,
        ``k``, ``power``, ...).

    Returns
    -------
    trees : list of Tree
        The trees as :func:`sylva.trees.prune_trees` returns them (sorted by
        DBH with ids 1..n; without pruning, the stems that survived merging
        with their own ids), with heights and point counts. Each keeps the
        ``extra`` of the stem it came from.
    catalog : Catalog
        The segmented tiles.

    Raises
    ------
    ValueError
        For bad parameters, stems without unique ids, or a plot that does
        not fit in memory one tile at a time.

    Warns
    -----
    UserWarning
        When crowns reach the edge of the widest buffer, or nodes were
        claimed by two tiles.
    """
    cat = _as_catalog(catalog)
    stems = list(stems)
    o = _origin(voxel_origin)
    merge_kw = _options(merge, "merge")
    if merge_kw is not None:
        merge_kw.setdefault("voxel_origin", o)
        if merge_kw["voxel_origin"] is not None:
            merge_kw["voxel_origin"] = _origin(merge_kw["voxel_origin"])
    prune_kw = _options(prune, "prune")
    params = dict(params)
    params["voxel_origin"] = o
    for name in ("buffer", "max_buffer", "edge_margin"):
        v = float(locals()[name])
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be a non-negative number of metres, got {v}")
    found, at_edge, paths, conflicts, info = _core.tiles_segment_trees(
        cat._core(), [t._to_core() for t in stems], str(out), _format(format), str(height_attr),
        params, merge_kw, float(percentile), prune_kw, float(buffer), float(max_buffer),
        float(edge_margin), str(attribute), _workers(workers))
    _record(info)
    edge = set(at_edge)
    out_trees = []
    for i, d in found:
        t = Tree._from_core(d)
        t.extra = dict(stems[i].extra)
        t.extra["crown_at_edge"] = t.tree_id in edge
        out_trees.append(t)
    if at_edge:
        warnings.warn(f"the crowns of trees {sorted(edge)} reach the edge of a {max_buffer:g} m buffer; "
                      "their labels may differ from the whole plot's. A wider max_buffer would "
                      "read their tiles further out.", stacklevel=2)
    if conflicts:
        warnings.warn(f"{conflicts} graph nodes were claimed by trees of two tiles; each took the "
                      "label of its own tile's tree. A wider buffer avoids it.", stacklevel=2)
    return out_trees, _written(list(paths), cat)


# ------------------------------------------------------------------ tree stores


class TreeStore:
    """Each tree's points of a segmented plot, one directory per tree.

    Written by :func:`split_trees`. A tree's points are those of every tile
    it touches, in catalogue order (the order the whole plot has them in),
    at full precision and with all their attributes. Per-point values of a
    tree (leaf / wood, say) can be kept beside it (:meth:`set_values`) and
    written back to the tiles (:func:`classify_leaf_wood` does so).

    Parameters
    ----------
    path
        The store's directory.

    Raises
    ------
    ValueError
        If ``path`` is not a tree store.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        names, entries = _core.tree_store_read(str(self.path))
        self.tiles: list[str] = list(names)
        self._entries = {int(e["tree_id"]): e for e in entries}

    def __repr__(self) -> str:
        return f"TreeStore({str(self.path)!r}, {len(self)} trees)"

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tree_id) -> bool:
        return int(tree_id) in self._entries

    @property
    def ids(self) -> list[int]:
        """Tree ids, ascending."""
        return sorted(self._entries)

    def n_points(self, tree_id: int) -> int:
        """Points of one tree."""
        return int(self._entry(tree_id)["n_points"])

    def bounds(self, tree_id: int) -> tuple[float, ...]:
        """``(xmin, ymin, zmin, xmax, ymax, zmax)`` of one tree's points."""
        return tuple(self._entry(tree_id)["bounds"])

    def tiles_of(self, tree_id: int) -> list[str]:
        """Names of the tiles holding points of one tree, in catalogue order."""
        return [self.tiles[t] for t, _ in self._entry(tree_id)["parts"]]

    def _entry(self, tree_id: int) -> dict:
        try:
            return self._entries[int(tree_id)]
        except KeyError:
            raise ValueError(f"the store has no tree {tree_id}") from None

    def read(self, tree_id: int) -> PointCloud:
        """One tree's points.

        Parameters
        ----------
        tree_id
            Which tree.

        Returns
        -------
        PointCloud
            In the order the whole plot has them, with every attribute.
        """
        self._entry(tree_id)
        xyz, attrs = _core.tree_store_tree(str(self.path), int(tree_id))
        return PointCloud(xyz, attrs)

    def values(self, tree_id: int, name: str) -> np.ndarray | None:
        """Values kept for one tree under ``name`` (None if there are none)."""
        self._entry(tree_id)
        return _core.tree_store_values(str(self.path), int(tree_id), str(name))

    def has_values(self, tree_id: int, name: str) -> bool:
        """Whether values are kept for one tree under ``name``."""
        return bool(_core.tree_store_has_values(str(self.path), int(tree_id), str(name)))

    def set_values(self, tree_id: int, name: str, values) -> None:
        """Keep a value per point of one tree (in :meth:`read` order) under ``name``.

        Raises
        ------
        ValueError
            For a count other than the tree's points, or a name that is not a
            plain word.
        """
        v = np.ascontiguousarray(values)
        if v.ndim != 1 or len(v) != self.n_points(tree_id):
            raise ValueError(f"tree {tree_id} has {self.n_points(tree_id)} points, got {v.shape} values")
        _core.tree_store_write_values(str(self.path), int(tree_id), str(name), v)

    def map(self, fn, ids=None, workers: int | None = None, label: str = "trees") -> dict:
        """``fn(tree_id, cloud)`` for every tree, a few trees at a time.

        Trees are taken largest first, each read when a worker is free, so
        at most ``workers`` trees are in memory; ``workers`` is lowered when
        that many of the largest tree would not fit in the memory budget
        (:mod:`sylva.util.limits`, :data:`BYTES_PER_TREE_POINT` per point). The
        work runs on threads: Sylva's computations release the interpreter,
        so threads run them in parallel.

        Parameters
        ----------
        fn
            Called with a tree's id and points; its result is kept.
        ids
            Trees to do; all if None.
        workers
            Trees at once; the number of CPUs if None.
        label
            For the progress bar.

        Returns
        -------
        dict
            ``{tree_id: fn(tree_id, cloud)}``, in id order.
        """
        todo = self.ids if ids is None else [int(i) for i in ids]
        for i in todo:
            self._entry(i)
        return _map_trees(todo, lambda i: fn(i, self.read(i)),
                          max((self.n_points(i) for i in todo), default=0), workers, label)


def _map_trees(ids: list[int], job, largest: int, workers: int | None, label: str) -> dict:
    """Run ``job(id)`` on threads, largest first, within the memory budget."""
    w = (os.cpu_count() or 1) if workers is None else _workers(workers)
    b = limits.budget()
    need = max(1, largest) * BYTES_PER_TREE_POINT
    if b is not None:
        if need > b:
            limits.check(largest, BYTES_PER_TREE_POINT, f"a tree of {largest:,} points", "fewer points per tree")
        w = max(1, min(w, b // need))
    w = max(1, min(w, len(ids)))
    out = {}
    with progress.task(label, len(ids)) as bar:
        if w == 1:
            for i in ids:
                out[i] = job(i)
                bar.update()
        else:
            with ThreadPoolExecutor(max_workers=w) as pool:
                futures = {pool.submit(job, i): i for i in ids}
                for f in as_completed(futures):
                    out[futures[f]] = f.result()
                    bar.update()
    return {i: out[i] for i in sorted(out)}


def split_trees(catalog: Catalog, out: str | Path, attribute: str = "tree_id",
                workers: int | None = None) -> TreeStore:
    """Write every tree's points to a :class:`TreeStore`, one tile at a time.

    Parameters
    ----------
    catalog
        Segmented tiles (:func:`segment_trees`).
    out
        Directory of the store; trees already in it are replaced.
    attribute
        Attribute holding each point's tree id; below 0 is no tree.
    workers
        Tiles at once; see :func:`sylva.als.tiles.voxel_downsample`.

    Returns
    -------
    TreeStore

    Raises
    ------
    ValueError
        For tiles without ``attribute``.
    """
    cat = _as_catalog(catalog)
    _, info = _core.tiles_split_trees(cat._core(), str(out), str(attribute), _workers(workers))
    _record(info)
    return TreeStore(out)


def read_tree(source, tree_id: int, attribute: str = "tree_id", bounds=None) -> PointCloud:
    """One tree's points, without loading the plot.

    Parameters
    ----------
    source
        A :class:`TreeStore` (or its directory), which reads only the tree's
        own files; or segmented tiles (a :class:`Catalog`), where every tile
        meeting ``bounds`` (all of them if None) is read and filtered.
    tree_id
        Which tree.
    attribute
        Label attribute of the tiles.
    bounds
        ``(xmin, ymin, xmax, ymax)`` known to hold the tree, for tiles.

    Returns
    -------
    PointCloud
        The tree's points in catalogue order, with their attributes.
    """
    if isinstance(source, TreeStore):
        return source.read(tree_id)
    if isinstance(source, (str, os.PathLike)) and (Path(source) / "index.tsv").exists():
        return TreeStore(source).read(tree_id)
    cat = _as_catalog(source)
    b = None if bounds is None else tuple(float(v) for v in bounds)
    xyz, attrs = _core.tiles_read_tree(cat._core(), int(tree_id), str(attribute), b)
    return PointCloud(xyz, attrs)


def _store(store) -> TreeStore:
    return store if isinstance(store, TreeStore) else TreeStore(store)


# ------------------------------------------------------------------ per-tree work


def classify_leaf_wood(store, catalog: Catalog, out: str | Path, min_points: int = 100,
                       voxel_size: float = 0.02, method: str = "gbs", attribute: str = "tree_id",
                       name: str = "wood", resume: bool = False, workers: int | None = None,
                       format: str | None = None, **wood_params) -> Catalog:
    """Leaf / wood of every tree, written to the tiles as ``wood``.

    Each tree of ``min_points`` points or more is classified on its own
    points (:func:`sylva.leaves.classify_leaf_wood`, which is defined on one
    tree), a few trees at a time; the tiles are then written to ``out`` with
    ``name`` (int8): 1 wood, 0 leaf, -1 for points of no tree or of a tree
    too small. The values are those of classifying ``plot[labels ==
    tree_id]`` of the whole plot.

    Parameters
    ----------
    store
        The trees (:func:`split_trees`).
    catalog
        The segmented tiles the store was split from.
    out
        Directory for the tiles with ``name``.
    min_points
        Smaller trees are left at -1.
    voxel_size, method, **wood_params
        As in :func:`sylva.leaves.classify_leaf_wood`.
    attribute
        Label attribute of the tiles.
    name
        Attribute written, and the name the values are kept under in the store.
    resume
        Keep values already in the store rather than classifying again.
    workers
        Trees, then tiles, at once.
    format
        ``"las"`` or ``"laz"``; that of each input tile if None.

    Returns
    -------
    Catalog
        The written tiles.
    """
    st = _store(store)
    cat = _as_catalog(catalog)
    ids = [i for i in st.ids if st.n_points(i) >= int(min_points)
           and not (resume and st.has_values(i, name))]

    def one(i, cloud):
        w = leaves.classify_leaf_wood(cloud, voxel_size=voxel_size, method=method, **dict(wood_params))
        st.set_values(i, name, np.asarray(w, dtype=np.int8))

    st.map(one, ids=ids, workers=workers, label="leaf / wood")
    paths, info = _core.tiles_write_back(cat._core(), str(st.path), str(out), str(attribute), str(name),
                                         -1.0, "int8", _format(format), _workers(workers))
    _record(info)
    return _written(list(paths), cat)


def _qsm_one(st: TreeStore, tree_id: int, stem: Tree | None, voxel_size, wood, buttress, min_points,
             height_attr, settings) -> dict:
    """One tree through the core of :func:`sylva.qsm.build_plot`."""
    cloud = st.read(tree_id)
    heights = cloud.attrs.get(height_attr)
    stems = [] if stem is None else [stem]
    d = _core.qsm_build_plot(cloud.xyz, np.full(len(cloud), int(tree_id), dtype=np.int64),
                             None if heights is None else np.ascontiguousarray(heights, dtype=float),
                             [int(s.tree_id) for s in stems], [(float(s.x), float(s.y)) for s in stems],
                             [float(getattr(s, "dbh", np.nan)) for s in stems],
                             float(voxel_size), bool(wood), bool(buttress), float(min_points), settings)
    d.pop("median_measured_length", None)
    return d


def build_qsms(store, stems=None, voxel_size: float = 0.01, wood: bool = True,
               buttress: bool = False, min_points: int = 2000, height_attr: str = "height",
               resume: bool = False, ids=None, workers: int | None = None,
               **params) -> qsm.PlotQSMs:
    """A QSM for every tree of a store, a few trees at a time.

    Each tree is modelled from its own points as :func:`sylva.qsm.build_plot`
    models it in the whole plot (thinning, the wood filter, the stem centre
    and DBH anchor from ``stems``, a buttress if asked for), so the models
    are those of ``build_plot(plot, labels, stems, ...)``. Trees run in
    parallel, each one's fit parallel inside; at most ``workers`` trees are
    in memory.

    Parameters
    ----------
    store
        The trees (:func:`split_trees`).
    stems
        The trees of :func:`segment_trees`, for each model's stem centre and
        DBH anchor.
    voxel_size, wood, buttress, min_points, height_attr
        As in :func:`sylva.qsm.build_plot`.
    resume
        Keep models already saved in the store (``qsm.pkl`` beside each tree)
        rather than fitting them again.
    ids
        Trees to model; every tree of the store if None.
    workers
        Trees at once; the number of CPUs if None, fewer if memory is short.
    **params
        Passed to :func:`sylva.qsm.build_qsm` (``stem_radius_cap``, ...).

    Returns
    -------
    PlotQSMs
        As :func:`sylva.qsm.build_plot` returns it.
    """
    st = _store(store)
    settings = qsm.plot._qsm_settings(params)
    by_id = {int(s.tree_id): s for s in (stems or [])}
    saved = {}

    def path(i):
        return st.path / f"tree_{i}" / "qsm.pkl"

    wanted = st.ids if ids is None else [int(i) for i in ids]
    for i in wanted:
        st._entry(i)
    if resume:
        for i in wanted:
            if path(i).exists():
                with open(path(i), "rb") as f:
                    saved[i] = pickle.load(f)

    def one(i, _cloud=None):
        d = _qsm_one(st, i, by_id.get(i), voxel_size, wood, buttress, min_points, height_attr, settings)
        tmp = path(i).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(d, f)
        tmp.replace(path(i))
        return d

    todo = [i for i in wanted if i not in saved]
    done = _map_trees(todo, one, max((st.n_points(i) for i in todo), default=0), workers, "fitting QSMs")
    results = {**saved, **done}
    models, bases, skipped, points, heights = {}, {}, {}, {}, {}
    for i in sorted(results):
        d = results[i]
        for t, c in d["models"]:
            models[t] = qsm.QSM(c)
        for t, b in d["buttresses"]:
            bases[t] = qsm.buttress._buttress(b)
        skipped.update(dict(d["skipped"]))
        points.update(dict(d["points"]))
        heights.update(dict(d["heights"]))
    out = qsm.PlotQSMs(models, bases, skipped)
    object.__setattr__(out, "_points", points)
    object.__setattr__(out, "_heights", heights)
    if models:
        share = float(np.median([m.metrics()["measured_length_fraction"] for m in models.values()]))
        if share < 0.1:
            warnings.warn(
                f"only {share:.0%} of the median model's length was fitted to points: "
                f"the cloud may be too sparse for bin_length={params.get('bin_length', 0.1)} m. "
                "Radii then come from the priors and run large; check measured_length in the table.",
                stacklevel=2)
    return out


def crown_metrics(store, height_attr: str = "height", crown_base_fraction: float = 0.1,
                  workers: int | None = None) -> dict[int, dict]:
    """:func:`sylva.trees.crown_metrics` for every tree of a store.

    Parameters
    ----------
    store
        The trees (:func:`split_trees`).
    height_attr
        Attribute holding heights.
    crown_base_fraction
        Density threshold for the crown base.
    workers
        Trees at once.

    Returns
    -------
    dict
        ``{tree_id: metrics}`` as :func:`sylva.trees.crown_metrics_all`
        gives it; trees with fewer than 4 points are missing.
    """
    st = _store(store)

    def one(i, cloud):
        return trees.crown_metrics(cloud, np.full(len(cloud), i, dtype=np.int64), i,
                                   height_attr=height_attr, crown_base_fraction=crown_base_fraction)

    return {i: m for i, m in st.map(one, workers=workers, label="crowns").items() if m}


# ------------------------------------------------------------------ workflow


@dataclass
class PlotRun:
    """What :func:`run_plot` made, and where.

    Attributes
    ----------
    out
        The run's directory.
    trees
        The trees, as :func:`segment_trees` returns them.
    table
        One row per tree: the tree, its crown metrics and its QSM's volume.
    qsms
        The models.
    catalogs
        ``{stage: Catalog}`` of the tiles of each stage.
    dtm
        The terrain model.
    store
        The per-tree store.
    timings
        Seconds per stage (0 for a stage taken from an earlier run).
    memory
        Most points held at once per stage (:attr:`RunInfo.max_points`).
    """

    out: Path
    trees: list[Tree]
    table: list[dict]
    qsms: qsm.PlotQSMs
    catalogs: dict[str, Catalog]
    dtm: Raster
    store: TreeStore
    timings: dict[str, float] = field(default_factory=dict)
    memory: dict[str, int] = field(default_factory=dict)


def _done(d: Path) -> bool:
    return (d / ".done").exists()


def _mark(d: Path, info: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / ".done").write_text(json.dumps(info or {}))


def _fresh(d: Path) -> Path:
    """A stage's directory, emptied of whatever an interrupted run left."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _positions(scans):
    """RiSCAN scan positions if ``scans`` names a project, else None."""
    from ..riscan import RiscanProject, read_riscan_project
    if isinstance(scans, RiscanProject):
        return scans.with_scans()
    if isinstance(scans, (str, os.PathLike)) and Path(scans).is_dir():
        return read_riscan_project(scans).with_scans()
    return None


def run_plot(scans, out: str | Path, transforms=None, use=None, bounds=None, plot=None,
             tile_size: float = 10.0, voxel_size: float = 0.02, read_options: dict | None = None,
             sor: dict | None = None, ground: dict | None = None, dtm_resolution: float = 0.5,
             ground_voxel: float = 0.05, stems: dict | None = None, merge=None, segment: dict | None = None,
             percentile: float = 99.0, prune=None, buffer: float = 20.0, max_buffer: float = 60.0,
             leaf_wood: dict | None = None, qsm_options: dict | None = None, qsm_files: bool = False,
             scale: float = 0.001, workers: int | None = None, log=print) -> PlotRun:
    """The plot workflow from registered scans to trees and QSMs, in tiles.

    Stages, each saved in its own directory of ``out`` and skipped when a
    rerun finds it complete (so an interrupted run resumes; per-tree stages
    resume tree by tree):

    ======================  ==============================================================
    ``scans/``              (RiSCAN projects only) each scan read, moved by its SOP, cropped to ``bounds`` and thinned
    ``tiles/``              :func:`sylva.als.tiles.from_scans`: tiles thinned to ``voxel_size`` on one grid
    ``sor/``                :func:`sylva.als.tiles.statistical_outlier_removal`
    ``ground/``             :func:`sylva.als.tiles.classify_ground` on a ``ground_voxel`` thinning; the DTM in ``dtm.npz`` (full precision, used for the heights) and ``dtm.asc``
    ``heights/``            :func:`sylva.als.tiles.normalize` with that DTM
    ``stems.pkl``           :func:`sylva.als.tiles.detect_stems` on the tiles inside ``plot``
    ``segmented/``          :func:`segment_trees`, with ``trees.pkl``
    ``trees/``              :func:`split_trees`, with per-tree leaf / wood and QSMs
    ``wood/``               :func:`classify_leaf_wood`
    ``qsm_table.csv``       :func:`build_qsms`, with ``qsm_models.pkl``
    ``trees.csv``           the trees with their crown metrics and QSM volumes
    ======================  ==============================================================

    Memory is bounded by the tile size, the buffers and ``workers``, not by
    the plot: no stage holds more than a few tiles with their buffers, or a
    few trees.

    Parameters
    ----------
    scans
        Registered scan files or point clouds in scan order (as
        :func:`sylva.als.tiles.from_scans` takes them), or a RiSCAN project (its
        directory, or a :class:`sylva.riscan.RiscanProject`), whose scans are
        read with their SOPs.
    out
        Directory of the run.
    transforms
        One 4x4 matrix (or None) per scan, applied first: the registration of
        the files, or for a project the corrections applied after each SOP.
    use
        One flag per scan; scans flagged False are left out.
    bounds
        ``(xmin, ymin, xmax, ymax)`` of everything kept: the plot and a
        margin for ground and buffers.
    plot
        ``(xmin, ymin, xmax, ymax)`` of the plot whose trees are wanted, on
        the tile grid (tiles inside it are used); all tiles if None.
    tile_size, voxel_size, scale
        Of :func:`sylva.als.tiles.from_scans`.
    read_options
        For a project: options of :meth:`sylva.riscan.ScanPosition.read`
        (``shot_stride``, ...).
    sor
        Keywords of :func:`sylva.als.tiles.statistical_outlier_removal` (default
        ``k=6, std_ratio=1``); False skips it.
    ground
        Keywords of :func:`sylva.als.tiles.classify_ground` (default CSF with
        ``cloth_resolution=0.5``, ``rigidness=2``, a 10 m buffer).
    dtm_resolution
        Cell size of the DTM (m).
    ground_voxel
        Thinning of the points ground is classified on (m).
    stems
        Keywords of :func:`sylva.als.tiles.detect_stems` (default
        ``min_arc_deg=130``).
    merge, prune
        As in :func:`segment_trees` (defaults ``{"voxel_size": 0.1}`` and
        ``{"min_height": 2.0}``).
    segment
        Keywords of :func:`sylva.trees.segment_trees` (default
        ``voxel_size=0.1``).
    percentile
        Height percentile of the trees.
    buffer, max_buffer
        Of :func:`segment_trees`.
    leaf_wood
        Keywords of :func:`classify_leaf_wood`; False skips it.
    qsm_options
        Keywords of :func:`build_qsms` (``stem_radius_cap``, ...); False skips it.
    qsm_files
        Also write each model's cylinders and mesh to ``qsm/``.
    workers
        Tiles or trees at once.
    log
        Called with a line of text per stage; None for silence.

    Returns
    -------
    PlotRun
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    say = log or (lambda _m: None)
    t0 = time.time()
    timings: dict[str, float] = {}
    memory: dict[str, int] = {}
    from . import tiles

    def stage(name, d, fn):
        if _done(d):
            timings[name] = 0.0
            say(f"[{time.time() - t0:7.0f} s] {name}: from the earlier run")
            return
        t = time.time()
        _record(None)
        info = fn()
        timings[name] = time.time() - t
        ri = tiles.last_run()
        if ri is not None:
            memory[name] = ri.max_points
        _mark(d, {"seconds": timings[name], "max_points": memory.get(name)})
        say(f"[{time.time() - t0:7.0f} s] {name}: {info}")

    # --------------------------------------------------------- scans to tiles
    positions = _positions(scans)
    n_scans = len(positions) if positions is not None else len(list(scans))
    flags = [True] * n_scans if use is None else [bool(u) for u in use]
    if len(flags) != n_scans:
        raise ValueError(f"{len(flags)} use flags for {n_scans} scans")
    mats = [None] * n_scans if transforms is None else list(transforms)
    if len(mats) != n_scans:
        raise ValueError(f"{len(mats)} transforms for {n_scans} scans")
    b = None if bounds is None else tuple(float(v) for v in bounds)
    if positions is not None:
        sdir = out / "scans"

        def export():
            sdir.mkdir(parents=True, exist_ok=True)
            with progress.task("reading scans", n_scans) as bar:
                for i, pos in enumerate(positions):
                    f = sdir / f"{i:03d}.laz"
                    if flags[i] and not f.exists():
                        pc = pos.read(**(read_options or {}))
                        if b is not None:
                            m = (pc.x >= b[0]) & (pc.x <= b[2]) & (pc.y >= b[1]) & (pc.y <= b[3])
                            pc = pc[m]
                        pc = filters.voxel_downsample(pc, voxel_size, origin=(0, 0, 0))
                        from .. import io
                        tmp = sdir / f"{i:03d}.tmp.laz"
                        io.write(pc, tmp)
                        tmp.replace(f)
                    bar.update()
            return f"{sum(flags)} scans read"

        stage("scans", sdir, export)
        sources = [sdir / f"{i:03d}.laz" for i in range(n_scans)]
    else:
        sources = list(scans)
    kept = [i for i in range(n_scans) if flags[i]]
    cat_dirs = {k: out / k for k in ("tiles", "sor", "ground_thin", "ground", "heights", "segmented",
                                     "wood")}

    def make_tiles():
        _fresh(cat_dirs["tiles"])
        c = tiles.from_scans([sources[i] for i in kept], cat_dirs["tiles"], tile_size=tile_size,
                             voxel_size=voxel_size, transforms=[mats[i] for i in kept], bounds=b,
                             scale=scale, workers=workers)
        return f"{c.n_points:,} points in {len(c)} tiles"

    stage("tiles", cat_dirs["tiles"], make_tiles)
    cats = {"tiles": catalog(cat_dirs["tiles"])}

    sor_kw = {"k": 6, "std_ratio": 1.0} if sor is None else (None if sor is False else dict(sor))
    if sor_kw is not None:
        def run_sor():
            _fresh(cat_dirs["sor"])
            c = tiles.statistical_outlier_removal(cats["tiles"], cat_dirs["sor"], workers=workers, **sor_kw)
            return f"{c.n_points:,} of {cats['tiles'].n_points:,} points kept"
        stage("sor", cat_dirs["sor"], run_sor)
        clean = catalog(cat_dirs["sor"])
    else:
        clean = cats["tiles"]
    cats["sor"] = clean

    ground_kw = {"method": "csf", "cloth_resolution": 0.5, "rigidness": 2, "buffer": 10.0}
    ground_kw.update(ground or {})
    dtm_path = out / "dtm.asc"
    # The DTM the heights are taken from, at full precision; dtm.asc (a few
    # decimals) is for reading elsewhere and would move heights by its rounding.
    dtm_full = out / "dtm.npz"

    def make_dtm(g):
        d = tiles.dtm(g, resolution=dtm_resolution, buffer=float(ground_kw.get("buffer", 10.0)),
                      workers=workers)
        np.savez(dtm_full, data=d.data, origin=np.array([d.xmin, d.ymin, d.resolution]))
        d.to_ascii_grid(dtm_path)
        return d

    def run_ground():
        _fresh(cat_dirs["ground_thin"])
        _fresh(cat_dirs["ground"])
        thin = tiles.voxel_downsample(clean, cat_dirs["ground_thin"], ground_voxel, workers=workers)
        g = tiles.classify_ground(thin, cat_dirs["ground"], workers=workers, **ground_kw)
        d = make_dtm(g)
        return f"DTM {d.data.shape[1]} x {d.data.shape[0]} cells at {dtm_resolution} m"

    stage("ground", cat_dirs["ground"], run_ground)
    if not dtm_full.exists():
        make_dtm(catalog(cat_dirs["ground"]))
    with np.load(dtm_full) as f:
        x0, y0, res = (float(v) for v in f["origin"])
        dtm = Raster(np.array(f["data"]), x0, y0, res)

    def run_heights():
        _fresh(cat_dirs["heights"])
        c = tiles.normalize(clean, cat_dirs["heights"], dtm=dtm, buffer=0.0, workers=workers)
        return f"{c.n_points:,} points with heights"

    stage("heights", cat_dirs["heights"], run_heights)
    heights = catalog(cat_dirs["heights"])
    cats["heights"] = heights
    if plot is not None:
        p = tuple(float(v) for v in plot)
        inside = [t.path for t in heights.tiles
                  if p[0] <= (t.bounds[0] + t.bounds[3]) / 2 <= p[2] and p[1] <= (t.bounds[1] + t.bounds[4]) / 2 <= p[3]]
        plot_cat = catalog(inside)
    else:
        plot_cat = heights

    stems_path = out / "stems.pkl"
    stem_kw = {"min_arc_deg": 130.0} if stems is None else dict(stems)

    def run_stems():
        found = tiles.detect_stems(plot_cat, workers=workers, **stem_kw)
        with open(stems_path, "wb") as f:
            pickle.dump(found, f)
        return f"{len(found)} stems"

    stage("stems", out / "stems.d", run_stems)
    with open(stems_path, "rb") as f:
        candidates = pickle.load(f)

    trees_path = out / "trees.pkl"
    merge_kw = {"voxel_size": 0.1} if merge is None else merge
    prune_kw = {"min_height": 2.0} if prune is None else prune
    seg_kw = {"voxel_size": 0.1} if segment is None else dict(segment)

    def run_segment():
        _fresh(cat_dirs["segmented"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            found, _ = segment_trees(plot_cat, candidates, cat_dirs["segmented"], buffer=buffer,
                                     max_buffer=max_buffer, merge=merge_kw, percentile=percentile,
                                     prune=prune_kw, workers=workers, **seg_kw)
        for w in caught:
            say(f"  {w.message}")
        with open(trees_path, "wb") as f:
            pickle.dump(found, f)
        return f"{len(candidates)} candidates -> {len(found)} trees"

    stage("segmentation", cat_dirs["segmented"], run_segment)
    with open(trees_path, "rb") as f:
        found = pickle.load(f)
    segmented = catalog(cat_dirs["segmented"])
    cats["segmented"] = segmented

    store_dir = out / "trees"

    def run_split():
        s = split_trees(segmented, store_dir, workers=workers)
        return f"{len(s)} trees stored"

    stage("split", out / "trees.d", run_split)
    store = TreeStore(store_dir)

    if leaf_wood is not False:
        def run_wood():
            _fresh(cat_dirs["wood"])
            c = classify_leaf_wood(store, segmented, cat_dirs["wood"], resume=True, workers=workers,
                                   **(leaf_wood or {}))
            return f"leaf / wood in {len(c)} tiles"
        stage("leaf/wood", cat_dirs["wood"], run_wood)
        cats["wood"] = catalog(cat_dirs["wood"])

    qsms = qsm.PlotQSMs({}, {}, {})
    if qsm_options is not False:
        q_kw = {"stem_radius_cap": 1.5} if qsm_options is None else dict(qsm_options)

        def run_qsm():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                m = build_qsms(store, found, resume=True, workers=workers, **q_kw)
            for w in caught:
                say(f"  {w.message}")
            with open(out / "qsm_models.pkl", "wb") as f:
                pickle.dump(m, f)
            m.to_csv(out / "qsm_table.csv")
            if qsm_files:
                m.write_cylinders(out / "qsm")
                m.write_meshes(out / "qsm")
            return f"{len(m)} trees modelled, {m.total_volume:.1f} m3"

        stage("QSMs", out / "qsm.d", run_qsm)
        with open(out / "qsm_models.pkl", "rb") as f:
            qsms = pickle.load(f)

    def run_table():
        crowns = crown_metrics(store, workers=workers)
        rows = []
        for t in found:
            row = {**t.as_dict(), **crowns.get(t.tree_id, {})}
            if t.tree_id in qsms.models:
                row["qsm_volume_m3"] = qsms.volume(t.tree_id)
            rows.append(row)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out / "trees.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with open(out / "trees_table.pkl", "wb") as f:
            pickle.dump(rows, f)
        return f"{len(rows)} trees in trees.csv"

    stage("table", out / "table.d", run_table)
    with open(out / "trees_table.pkl", "rb") as f:
        table = pickle.load(f)
    return PlotRun(out, found, table, qsms, cats, dtm, store, timings, memory)
