# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""QSMs of every tree in a plot."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ._common import _check_weights
from .buttress import Buttress, _buttress
from .mesh import _faces
from .model import QSM, build_qsm


@dataclass
class PlotQSMs:
    """Every tree of a plot modelled; build with :func:`build_plot`.

    Attributes
    ----------
    models
        ``{tree_id: QSM}`` for the trees that were fitted.
    buttresses
        ``{tree_id: Buttress}`` where a buttress was found and meshed.
    skipped
        ``{tree_id: reason}`` for the trees that were not modelled: too few
        points, or the message of the fit that failed.
    """

    models: dict[int, QSM]
    buttresses: dict[int, "Buttress"]
    skipped: dict[int, str]

    def __len__(self) -> int:
        return len(self.models)

    def volume(self, tree_id: int) -> float:
        """Wood volume of one tree (m³), buttress included where there is one.

        Parameters
        ----------
        tree_id
            Which tree.

        Returns
        -------
        float
        """
        return _core.plot_total_volume([self._entry(tree_id, self.models[tree_id])])

    @property
    def total_volume(self) -> float:
        """Wood volume of the whole plot (m³)."""
        return _core.plot_total_volume(self._entries())

    def _entry(self, t: int, m: QSM) -> tuple:
        """One tree as the core reads a plot."""
        b = self.buttresses.get(t)
        if b is not None:
            b = (np.ascontiguousarray(b.vertices, dtype=float), _faces(b.faces), float(b.volume), float(b.top),
                 float(b.top_z))
        points = self._points.get(t)
        height = self._heights.get(t)
        return (int(t), np.ascontiguousarray(m.cylinders, dtype=float), None if points is None else int(points),
                None if height is None else float(height), b)

    def _entries(self) -> list:
        return [self._entry(t, m) for t, m in self.models.items()]

    def table(self) -> list[dict]:
        """One row per tree, ready for a CSV.

        Returns
        -------
        list of dict
            ``tree_id``, ``points``, ``volume_m3``, ``dbh_m``, ``height_m``,
            ``n_cylinders``, ``measured_volume`` and ``measured_length`` (the
            share of the model that was fitted to points rather than taken
            from the taper and pipe-model priors), and ``buttress_m3`` /
            ``buttress_top_m`` (blank without a buttress).
        """
        keys = ("tree_id", "points", "volume_m3", "dbh_m", "height_m", "n_cylinders", "measured_volume",
                "measured_length", "buttress_m3", "buttress_top_m")
        return [{k: "" if v is None else v for k, v in zip(keys, row, strict=True)}
                for row in _core.plot_table(self._entries())]

    def to_csv(self, path: str | Path) -> None:
        """Write :meth:`table` as a CSV.

        Parameters
        ----------
        path
            Output file.
        """
        _core.plot_write_csv(str(path), self._entries())

    def write_meshes(self, directory: str | Path, fmt: str = "ply", sides: int = 12,
                     contiguous: bool = True, prefix: str = "tree") -> list[Path]:
        """Write a surface mesh per tree into ``directory``.

        A tree with a buttress is written fused (:meth:`Buttress.fuse`), so
        the flanged base and the cylinders above it come out as one file;
        every other tree is its cylinder mesh.

        Parameters
        ----------
        directory
            Created if it does not exist.
        fmt : {"ply", "obj"}
            PLY is binary and carries face colours; OBJ is text and keeps the
            buttress and the wood as named objects.
        sides
            Facets around each cylinder.
        contiguous
            One continuous tube per branch (see :meth:`QSM.mesh`).
        prefix
            File name stem; files are ``<prefix><tree_id>.<fmt>``.

        Returns
        -------
        list of pathlib.Path
            The files written, in tree order.

        Raises
        ------
        ValueError
            For an unknown format.
        """
        d = Path(directory)
        written = _core.plot_write_meshes(str(d), self._entries(), str(fmt), int(sides), bool(contiguous), str(prefix))
        return [d / Path(p).name for p in written]

    def write_cylinders(self, directory: str | Path, prefix: str = "tree") -> None:
        """Write one cylinder CSV per tree into ``directory``.

        Parameters
        ----------
        directory
            Created if it does not exist.
        prefix
            File name stem; files are ``<prefix><tree_id>.csv``.
        """
        _core.plot_write_cylinders(str(directory), self._entries(), str(prefix))

    #: filled in by build_plot
    _points: dict = None
    _heights: dict = None

    def __post_init__(self) -> None:
        if self._points is None:
            object.__setattr__(self, "_points", {})
        if self._heights is None:
            object.__setattr__(self, "_heights", {})


def build_plot(cloud: PointCloud, labels, stems=None, voxel_size: float = 0.01,
               wood=True, buttress: bool = False, min_points: int = 2000,
               height_attr: str = "height", **params) -> PlotQSMs:
    """A QSM for every tree of a segmented plot.

    The loop that :func:`build_qsm` needs around it: each tree's points are
    taken from ``labels``, thinned, put through the wood filter and fitted,
    and a tree that cannot be fitted is recorded rather than raising. Progress
    is reported (:mod:`sylva.util.progress`), so a plot of a few hundred trees is
    not silent.

    Parameters
    ----------
    cloud
        The whole plot, height-normalised (needed for ``buttress``).
    labels
        Tree id per point, as :func:`sylva.trees.segment_trees` returns;
        anything below 0 is not part of a tree.
    stems
        The detected trees, used for the stem centre each model is built
        around, and their DBH, which anchors each model's base radius
        (``base_radius = dbh / 2``) unless ``base_radius`` is given. Without
        that anchor, a small tree with a leafy crown can take its trunk
        radius from a foliage clump. Without stems the centre is the middle
        of the tree's own points between 0.5 and 1.5 m.
    voxel_size
        Thin each tree to this spacing first (m); 0 keeps every point.
    wood : bool, array or str
        Where each tree's wood comes from. True (default) runs
        :func:`wood_points` on each tree; False fits every point (clouds that
        are wood already). A boolean array with one value per point gives the
        wood directly (True is wood; for instance from
        :func:`sylva.leaves.classify_leaf_wood`): each tree is fitted on its
        wood points. A float array of weights in [0, 1] (a wood confidence,
        ``classify_leaf_wood(..., return_scores=True)``) fits each tree on all its points
        with those weights (``build_qsm(weights=)``). A string names a cloud
        attribute: integer or boolean values are labels (1 or True is wood, 0
        and -1 are not), floating-point values are weights. A tree's wood
        points are selected before they are thinned; with weights each point
        kept by the thinning keeps its own weight.
    buttress
        Look for a buttress on each tree (:func:`sylva.trees.detect_buttress`)
        and mesh it, so the volume of a flanged base is not left to the
        cylinders. Needs ``height_attr`` on the cloud.
    min_points
        Trees with fewer points than this are skipped.
    height_attr
        Attribute holding height above ground.
    **params
        Passed to :func:`build_qsm`.

    Returns
    -------
    PlotQSMs
        The models, any buttresses, and why a tree was skipped.

    Examples
    --------
    >>> labels = trees.segment_trees(cloud, stems)          # doctest: +SKIP
    >>> plot = qsm.build_plot(cloud, labels, stems)         # doctest: +SKIP
    >>> plot.total_volume, len(plot), plot.skipped          # doctest: +SKIP
    >>> plot.to_csv("trees.csv"); plot.write_cylinders("qsms/")   # doctest: +SKIP

    Notes
    -----
    Volumes are the cylinders' own unless a buttress was meshed, in which
    case :meth:`PlotQSMs.volume` is the mesh below its top plus the cylinders
    above it.

    A QSM needs points on the stem surface: with roughly 1 cm spacing the
    default 0.1 m shells hold plenty, but a cloud thinned to 3-5 cm leaves
    most shells with too few, and those cylinders take their radius from the
    taper and pipe-model priors instead. That runs large - on one 20 m
    savanna tree, 0.34 m DBH at full resolution against 1.10 m at 5 cm - so
    ``build_plot`` warns when the median model was hardly fitted at all, and
    ``measured_length`` in :meth:`PlotQSMs.table` says so per tree.
    """
    labels = np.asarray(labels)
    if len(labels) != len(cloud):
        raise ValueError("labels must have one value per point")
    wood_flag, wood_labels, wood_weights = _plot_wood(cloud, wood)
    if labels.dtype.kind == "f":
        whole = np.isnan(labels) | (labels == np.trunc(labels))
        if not whole.all():
            raise ValueError("labels must be whole numbers")
        labels = np.where(np.isnan(labels), -1, labels)
    qsm_params = _qsm_settings(params)
    stems = list(stems) if stems is not None else []
    heights = cloud.attrs.get(height_attr)
    d = _core.qsm_build_plot(cloud.xyz, np.ascontiguousarray(labels, dtype=np.int64),
                             None if heights is None else np.ascontiguousarray(heights, dtype=float),
                             [int(s.tree_id) for s in stems], [(float(s.x), float(s.y)) for s in stems],
                             [float(getattr(s, "dbh", np.nan)) for s in stems],
                             float(voxel_size), wood_flag, bool(buttress), float(min_points), qsm_params,
                             wood_labels=wood_labels, wood_weights=wood_weights)
    out = PlotQSMs({t: QSM(c) for t, c in d["models"]}, {t: _buttress(b) for t, b in d["buttresses"]},
                   dict(d["skipped"]))
    object.__setattr__(out, "_points", dict(d["points"]))
    object.__setattr__(out, "_heights", dict(d["heights"]))
    share = d["median_measured_length"]
    if share is not None:
        # A model whose cylinders were never fitted to points is the taper and
        # pipe-model priors talking, and those inflate. The usual cause is a
        # cloud too sparse for the shell width.
        if share < 0.1:
            warnings.warn(
                f"only {share:.0%} of the median model's length was fitted to points: "
                f"the cloud may be too sparse for bin_length={params.get('bin_length', 0.1)} m. "
                "Radii then come from the priors and run large; check measured_length in the table.",
                stacklevel=2,
            )
    return out


def _plot_wood(cloud: PointCloud, wood):
    """``build_plot``'s ``wood``: the filter switch, or per-point labels or weights."""
    if isinstance(wood, (bool, np.bool_)):
        return bool(wood), None, None
    if isinstance(wood, str):
        if wood not in cloud.attrs:
            raise ValueError(f"no attribute {wood!r} on the cloud (it has {sorted(cloud.attrs)})")
        values = np.asarray(cloud.attrs[wood])
        what = f"attribute {wood!r}"
    else:
        values = np.asarray(wood)
        what = "wood"
    if values.ndim != 1 or len(values) != len(cloud):
        raise ValueError(f"{what} must have one value per point ({values.size} for {len(cloud)} points)")
    if values.dtype.kind == "b":
        return True, np.ascontiguousarray(values), None
    if values.dtype.kind in "iu":
        if not np.isin(values, (-1, 0, 1)).all():
            raise ValueError(f"{what} must hold wood labels 1 (wood), 0 or -1 (not wood)")
        return True, np.ascontiguousarray(values == 1), None
    if values.dtype.kind == "f":
        return True, None, _check_weights(values, len(cloud), what)
    raise ValueError(f"{what} must be True/False, per-point labels or per-point weights")


def _qsm_settings(params: dict) -> dict:
    """Every :func:`build_qsm` setting, its default unless ``params`` sets it."""
    sig = inspect.signature(build_qsm)
    for k in ("cloud", "base_xy"):
        if k in params:
            raise TypeError(f"build_qsm() got multiple values for argument '{k}'")
    if "weights" in params:
        raise TypeError("build_plot() takes per-point weights as wood=, not weights=")
    sig.bind(None, **params)  # a TypeError for a setting build_qsm does not take
    out = {k: p.default for k, p in sig.parameters.items() if k not in ("cloud", "base_xy", "weights")}
    out.update(params)
    return out
