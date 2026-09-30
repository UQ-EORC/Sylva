# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ray-traced voxel grids traced and stored block by block.

:func:`sylva.voxels.ray_voxelize` with ``block_size=`` cuts the grid into
blocks and traces a few blocks at a time, so the memory taken while tracing
is set by ``max_memory`` rather than by the size of the grid. Each pass over
the blocks streams every pulse once and traces it into the blocks its line
passes through; a pulse is always walked through the whole grid, so every
voxel receives exactly what a whole-grid trace gives it (see the guide for
the exactness argument). With ``out=`` the finished blocks are written to a
directory as they complete, and the grid is returned as a
:class:`BlockedVoxelGrid`, which reads blocks back on demand.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from . import _core
from .voxels import RayVoxelGrid, STATES

__all__ = ["BlockedVoxelGrid", "open_blocked"]


class BlockedVoxelGrid:
    """A ray-traced voxel grid kept on disk block by block.

    Made by :func:`sylva.voxels.ray_voxelize` with ``out=``, or opened with
    :func:`open_blocked`. The directory holds ``grid.json`` (placement,
    block size, tracing options), ``ground_height.f64`` when a DTM was given,
    and one Parquet file per block with a row per voxel and a column per raw
    field (blocks no pulse reached are not written and read as zeros).

    Any block or box of voxels can be read as a :class:`~sylva.voxels.RayVoxelGrid`
    with every derived field. Layer profiles, the occlusion summaries,
    :func:`~sylva.voxels.tree_sampling` and :meth:`write` run over the grid
    one slab of blocks at a time; whole arrays by name (``grid["pad_fpl"]``)
    are assembled slab by slab and need the memory of one array.
    """

    def __init__(self, core) -> None:
        self._core = core
        self.block_stats: dict | None = None

    @property
    def path(self) -> Path:
        """Directory of the grid."""
        return Path(self._core.path)

    @property
    def origin(self) -> np.ndarray:
        """Minimum corner of the grid, ``(x, y, z)``."""
        return self._core.origin

    @property
    def voxel_size(self) -> float:
        """Voxel edge (m)."""
        return self._core.voxel_size

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid size as ``(nx, ny, nz)``; arrays are ``(nz, ny, nx)``."""
        return self._core.shape

    @property
    def block_size(self) -> tuple[int, int, int]:
        """Voxels per block along x, y and z (the last block of an axis may be shorter)."""
        return self._core.block_size

    @property
    def n_blocks(self) -> tuple[int, int, int]:
        """Blocks along x, y and z."""
        return self._core.n_blocks

    @property
    def fields(self) -> list[str]:
        """Names of the raw fields."""
        return self._core.field_names()

    @property
    def metrics(self) -> list[str]:
        """Names of the derived metrics, as for :class:`~sylva.voxels.RayVoxelGrid`."""
        return self._core.metric_names()

    def __repr__(self) -> str:
        nx, ny, nz = self.shape
        bx, by, bz = self.n_blocks
        return f"BlockedVoxelGrid({nx}x{ny}x{nz} @ {self.voxel_size:g} m in {bx}x{by}x{bz} blocks, {str(self.path)!r})"

    def __getitem__(self, name: str) -> np.ndarray:
        get = self._core.field if name in self.fields else self._core.metric
        return get(name)

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except ValueError as e:
            raise AttributeError(name) from e

    def blocks_present(self) -> list[tuple[int, int, int]]:
        """Blocks written to disk, ``(bx, by, bz)``; the others hold no pulse."""
        return self._core.blocks_present()

    def block(self, index: tuple[int, int, int]) -> RayVoxelGrid:
        """One block as a grid of its own.

        Parameters
        ----------
        index
            ``(bx, by, bz)``, each below :attr:`n_blocks`.

        Returns
        -------
        RayVoxelGrid
            With its origin at the block's corner. Voxel centres are computed
            from that origin, so they can differ from the whole grid's in the
            last bit.
        """
        return RayVoxelGrid(self._core.block(tuple(int(v) for v in index)))

    def blocks(self) -> Iterator[tuple[tuple[int, int, int], RayVoxelGrid]]:
        """Every block, one at a time, as ``((bx, by, bz), grid)``, z slowest."""
        nx, ny, nz = self.n_blocks
        for bz in range(nz):
            for by in range(ny):
                for bx in range(nx):
                    yield (bx, by, bz), self.block((bx, by, bz))

    def read(self, lo, hi) -> RayVoxelGrid:
        """Voxels ``lo`` to ``hi`` as a grid of their own.

        Parameters
        ----------
        lo, hi
            ``(i, j, k)`` voxel indices; ``hi`` is exclusive.

        Returns
        -------
        RayVoxelGrid
        """
        lo = tuple(int(v) for v in lo)
        hi = tuple(int(v) for v in hi)
        if len(lo) != 3 or len(hi) != 3:
            raise ValueError("lo and hi are (i, j, k) voxel indices")
        return RayVoxelGrid(self._core.read_box(lo, hi))

    def to_grid(self) -> RayVoxelGrid:
        """The whole grid in memory (about 0.13 kB per voxel)."""
        return RayVoxelGrid(self._core.to_grid())

    @property
    def observed(self) -> np.ndarray:
        """Boolean ``(nz, ny, nx)``: voxels crossed by a pulse before its last echo."""
        return self["state"] >= STATES["empty"]

    def profile(self, name: str = "pad_fpl", min_beams: int = 1) -> np.ndarray:
        """Mean of a field or metric per layer; see :meth:`sylva.voxels.RayVoxelGrid.profile`."""
        return self._core.profile(name, float(min_beams))

    def occlusion_profile(self, min_height: float = 0.0, max_height: float | None = None) -> dict:
        """What the scan saw of the canopy space; see :meth:`sylva.voxels.RayVoxelGrid.occlusion_profile`."""
        return self._core.occlusion_profile(float(min_height), None if max_height is None else float(max_height))

    def observed_map(self, min_height: float = 0.0, max_height: float | None = None) -> np.ndarray:
        """Observed share of each column; see :meth:`sylva.voxels.RayVoxelGrid.observed_map`."""
        return self._core.observed_map(float(min_height), None if max_height is None else float(max_height))

    def write(self, path: str | Path, format: str | None = None, include_unobserved: bool = False,
              filled_only: bool = False) -> int:
        """Write the grid as one ``.vox`` or text file, a slab of blocks at a time.

        The rows are those :meth:`sylva.voxels.RayVoxelGrid.write` gives for
        the whole grid; see there for the parameters.

        Returns
        -------
        int
            Number of voxels written.
        """
        if format is None:
            format = "vox" if Path(path).suffix.lower() == ".vox" else "text"
        return self._core.write(str(path), format, include_unobserved, filled_only)


def open_blocked(path: str | Path) -> BlockedVoxelGrid:
    """Open a grid written by :func:`sylva.voxels.ray_voxelize` with ``out=``.

    Parameters
    ----------
    path
        The grid's directory.

    Returns
    -------
    BlockedVoxelGrid

    Raises
    ------
    OSError
        If the directory holds no complete blocked grid.
    """
    return BlockedVoxelGrid(_core.open_blocked_voxels(str(path)))


def _block_size(block_size) -> tuple[int, int, int]:
    b = np.atleast_1d(np.asarray(block_size))
    if b.size == 1:
        b = np.repeat(b, 3)
    if b.shape != (3,) or not np.issubdtype(b.dtype, np.integer) or np.any(b < 1):
        raise ValueError(f"block_size must be a positive number of voxels or three of them, got {block_size!r}")
    return tuple(int(v) for v in b)


def _trace(source, is_path: bool, block_size, max_memory, workers, out, kw) -> RayVoxelGrid | BlockedVoxelGrid:
    """Run the blocked trace (called by :func:`sylva.voxels.ray_voxelize`)."""
    block = _block_size(64 if block_size is None else block_size)
    if max_memory is not None:
        max_memory = float(max_memory)
        if not max_memory > 0:
            raise ValueError("max_memory must be a positive number of GB")
        max_memory *= 1e9
    if workers is None:
        workers = 0
    elif int(workers) != workers or workers < 1:
        raise ValueError("workers must be a positive integer")
    src = {"path": str(source)} if is_path else {"shots": source}
    core, stats = _core.ray_voxelize_blocks(**src, block=block, max_memory=max_memory, workers=int(workers),
                                            out=None if out is None else str(out), **kw)
    grid = BlockedVoxelGrid(core) if out is not None else RayVoxelGrid(core)
    grid.block_stats = stats
    return grid
