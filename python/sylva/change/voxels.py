# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Occupancy change between two ray-traced voxel grids.

A voxel that holds no echoes in the later epoch has only lost its contents
if pulses went through it; if none did, the later scan simply did not see
it. :func:`occupancy` uses the ray-traced grids of
:func:`sylva.voxels.ray_voxelize` to tell the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .. import _core
from ..voxels import RayVoxelGrid

__all__ = ["OCCUPANCY_CLASSES", "Occupancy", "occupancy"]

#: Codes of :attr:`Occupancy.classes`.
OCCUPANCY_CLASSES = {"unobserved": 0, "stable_empty": 1, "stable_occupied": 2, "gained": 3, "lost": 4}


@dataclass
class Occupancy:
    """Voxel occupancy change between two epochs, from :func:`occupancy`.

    Arrays are ``(nz, ny, nx)`` like those of
    :class:`~sylva.voxels.RayVoxelGrid`.

    Attributes
    ----------
    classes
        ``uint8`` class code per voxel (:data:`OCCUPANCY_CLASSES`).
    pad_a, pad_b
        Plant area density of each epoch (m² m⁻³), the ``pad`` metric of the
        two grids.
    pad_change
        ``pad_b - pad_a`` where both epochs sent at least ``min_pulses``
        pulses into the voxel, NaN elsewhere.
    layers
        Per voxel layer, bottom first: ``z`` (layer centre, grid frame), the
        voxel count of every class (``unobserved``, ``stable_empty``,
        ``stable_occupied``, ``gained``, ``lost``), ``n_compared`` (voxels in
        the PAD comparison), and the mean ``pad_a``, ``pad_b`` and their
        difference ``pad_change`` over those voxels (NaN where there are
        none). Multiply ``pad_change`` by the voxel size and sum for the
        change in plant area index over the commonly observed space.
    origin
        Minimum corner of the grids.
    voxel_size
        Voxel edge (m).
    min_pulses, min_hits, alpha
        Criteria used.
    """

    classes: np.ndarray
    pad_a: np.ndarray
    pad_b: np.ndarray
    pad_change: np.ndarray
    layers: dict[str, np.ndarray]
    origin: np.ndarray
    voxel_size: float
    min_pulses: int
    min_hits: int
    alpha: float

    def __repr__(self) -> str:
        nz, ny, nx = self.classes.shape
        counts = ", ".join(f"{k}={v:,}" for k, v in self.counts().items())
        return f"Occupancy({nx}x{ny}x{nz} @ {self.voxel_size:g} m: {counts})"

    def mask(self, name: str) -> np.ndarray:
        """Boolean ``(nz, ny, nx)`` mask of one class.

        Parameters
        ----------
        name
            A key of :data:`OCCUPANCY_CLASSES`, e.g. ``"lost"``.

        Returns
        -------
        numpy.ndarray
        """
        if name not in OCCUPANCY_CLASSES:
            raise ValueError(f"unknown class {name!r}; expected one of {', '.join(map(repr, OCCUPANCY_CLASSES))}")
        return self.classes == OCCUPANCY_CLASSES[name]

    def counts(self) -> dict[str, int]:
        """Number of voxels in each class.

        Returns
        -------
        dict
            ``{class name: count}``.
        """
        n = np.bincount(self.classes.ravel(), minlength=len(OCCUPANCY_CLASSES))
        return {k: int(n[v]) for k, v in OCCUPANCY_CLASSES.items()}

    def volume(self, name: str) -> float:
        """Volume (m³) of the voxels in one class.

        Parameters
        ----------
        name
            A key of :data:`OCCUPANCY_CLASSES`.

        Returns
        -------
        float
        """
        return float(self.mask(name).sum()) * self.voxel_size ** 3

    def centers(self, name: str) -> np.ndarray:
        """Centres of the voxels in one class, e.g. to plot what was lost.

        Parameters
        ----------
        name
            A key of :data:`OCCUPANCY_CLASSES`.

        Returns
        -------
        numpy.ndarray
            ``(K, 3)`` coordinates.
        """
        k, j, i = np.nonzero(self.mask(name))
        ijk = np.column_stack([i, j, k]).astype(np.float64)
        return np.asarray(self.origin, dtype=np.float64) + (ijk + 0.5) * self.voxel_size


def occupancy(grid_a: RayVoxelGrid, grid_b: RayVoxelGrid, pad: str = "pad_fpl", min_pulses: int = 10,
              min_hits: int = 1, alpha: float = 0.05) -> Occupancy:
    """Voxel occupancy change between two epochs of a plot.

    In each epoch a voxel is *occupied* when it holds at least ``min_hits``
    echoes, *empty* when it holds none and at least ``min_pulses`` pulses
    entered it before their last echo, and *not observed* otherwise: no pulse
    reached it, only pulses already stopped did (``occluded``), or too few
    passed to call it empty. A voxel occupied in one epoch and empty in the
    other is only called ``lost`` (or ``gained``) if the empty epoch sent
    enough pulses to have found its contents: if the occupied epoch saw a
    share ``p`` of the pulses entering the voxel stopped there, ``n`` pulses
    all pass through contents as dense with probability ``(1 - p) ** n``,
    which must not exceed ``alpha``. Sparse contents grazed by a few pulses
    are thus not reported as change. The classes are then

    | Class | Epoch a | Epoch b |
    |---|---|---|
    | ``stable_empty`` | empty | empty |
    | ``stable_occupied`` | occupied | occupied |
    | ``gained`` | empty | occupied |
    | ``lost`` | occupied | empty |
    | ``unobserved`` | not observed in one or both epochs, or too few pulses to tell | |

    so foliage the later scan could not see is ``unobserved``, never
    ``lost``. Plant area density change is reported per voxel and per layer
    over the voxels that both epochs sampled with at least ``min_pulses``
    pulses, so that the layer means compare like with like.

    Parameters
    ----------
    grid_a, grid_b
        Earlier and later grids from :func:`sylva.voxels.ray_voxelize`, of
        pulses registered to one frame and traced with the same
        ``voxel_size`` and ``bounds``. ``occlusion=True`` is not needed for
        the classes (occluded and unreached voxels are both unobserved) but
        lets :meth:`~sylva.voxels.RayVoxelGrid.occlusion_profile` say which.
    pad
        Plant area density metric compared, e.g. ``"pad_fpl"`` or
        ``"pad_ppl"``.
    min_pulses
        Fewest pulses entering a voxel for it to count as empty, and for its
        density to be compared.
    min_hits
        Fewest echoes for a voxel to count as occupied; 2 or more discounts
        isolated noise echoes.
    alpha
        Largest probability that the empty epoch missed contents that were
        still there, for a voxel to be called ``lost`` or ``gained``
        (1 disables the test).

    Returns
    -------
    Occupancy

    Raises
    ------
    ValueError
        If the grids differ in origin, voxel size or shape, if ``pad`` is not
        a metric of the grids, if ``min_pulses`` or ``min_hits`` is below 1,
        or if ``alpha`` is not in (0, 1].

    Examples
    --------
    >>> ga = sylva.voxels.ray_voxelize(shots_2020, 0.25, bounds)       # doctest: +SKIP
    >>> gb = sylva.voxels.ray_voxelize(shots_2024, 0.25, bounds)       # doctest: +SKIP
    >>> occ = sylva.change.occupancy(ga, gb, min_pulses=10)            # doctest: +SKIP
    >>> occ.volume("lost"), occ.layers["pad_change"]                   # doctest: +SKIP
    """
    for name, g in (("grid_a", grid_a), ("grid_b", grid_b)):
        if not isinstance(g, RayVoxelGrid):
            raise ValueError(f"{name} must be a RayVoxelGrid from sylva.voxels.ray_voxelize, got {type(g).__name__}")
    for name, v in (("min_pulses", min_pulses), ("min_hits", min_hits)):
        if isinstance(v, bool) or not isinstance(v, Integral) or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    r = _core.change_occupancy(grid_a._core, grid_b._core, pad, int(min_pulses), int(min_hits), float(alpha))
    return Occupancy(r["class"], r["pad_a"], r["pad_b"], r["pad_change"], r["layers"],
                     np.asarray(grid_a.origin, dtype=np.float64), float(grid_a.voxel_size), int(min_pulses),
                     int(min_hits), float(alpha))
