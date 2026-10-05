# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The chunk engine: run a function over every buffered chunk of a catalogue."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..pointcloud import PointCloud
from ..raster import Raster
from ..util import progress
from .catalogue import Catalog, Chunk, _as_catalog, _format, _workers, _written

_BUFFER_ATTR = "buffer"


def _strip_buffer(cloud: PointCloud, chunk: Chunk) -> PointCloud:
    """The core points of a function's output, without the buffer flag."""
    if _BUFFER_ATTR in cloud.attrs:
        keep = ~np.asarray(cloud.attrs[_BUFFER_ATTR], dtype=bool)
        return cloud[keep].without(_BUFFER_ATTR)
    x0, y0, x1, y1 = chunk.core
    inside = (cloud.x >= x0) & (cloud.y >= y0)
    inside &= (cloud.x <= x1) & (cloud.y <= y1) if chunk.own is not None else (cloud.x < x1) & (cloud.y < y1)
    return cloud[inside]


def apply(catalog: Catalog, fn: Callable, chunk_size: float | None = None, buffer: float = 20.0,
          workers: int | None = None, out: str | Path | None = None, format: str | None = None,
          origin=None):
    """Run a function over a catalogue, chunk by chunk, with buffers.

    Each chunk is read (only the files overlapping it, only the points in
    its buffered box) and handed to ``fn`` as ``fn(cloud, chunk)``.
    ``cloud`` carries a boolean ``buffer`` attribute, True for the points
    around the chunk that belong to its neighbours; ``chunk`` is the
    :class:`Chunk`. Chunks with no points of their own are skipped.

    Parameters
    ----------
    catalog
        From :func:`catalog` (or anything :func:`catalog` accepts).
    fn
        ``fn(cloud, chunk)`` returning a :class:`~sylva.PointCloud`, a
        :class:`~sylva.Raster`, None or anything else. A point cloud should
        keep the ``buffer`` attribute (any subset of ``cloud`` does), so
        that the buffer can be removed; without it the points outside the
        chunk's core are dropped by position.
    chunk_size
        Side of square chunks (m); None for one chunk per tile.
    buffer
        Width (m) of the band of neighbouring points around each chunk.
        Make it at least as wide as whatever ``fn`` looks across (a
        filter's window, a crown's radius).
    workers
        Chunks processed at once; the number of CPUs if None. Fewer are
        run when that many chunks would not fit in the memory budget
        (:mod:`sylva.util.limits`, assuming about 256 bytes per point). The
        result does not depend on it.
    out
        Directory for point outputs: each chunk's core points are written
        to ``out/<chunk name>.<ext>`` in the point format, scale and CRS of
        its tile, instead of being returned.
    format
        ``"las"`` or ``"laz"`` for the files in ``out``; that of each
        chunk's tile if None.
    origin
        Grid corner for ``chunk_size``; see :meth:`Catalog.chunks`.

    Returns
    -------
    PointCloud, Catalog, Raster or list
        What ``fn`` returned, assembled: point clouds are stacked in chunk
        order without their buffers (or, with ``out``, the :class:`Catalog`
        of the files written); rasters, which must share a resolution and
        the catalogue grid's alignment (corner a whole number of cells from
        the catalogue's minimum snapped down to the resolution), are joined
        into one, each cell from the chunk whose core holds its centre; any
        other results come back as a list in chunk order, None for skipped
        chunks.

    Raises
    ------
    ValueError
        For bad arguments, a catalogue with missing files, a chunk too large
        for the memory budget, or rasters that do not line up. Whatever
        ``fn`` raises is raised as it is, and no new chunks are started.

    Examples
    --------
    Any single-cloud function can run this way. A canopy height model from
    tiles already normalised, on the catalogue grid (``bounds`` from the
    chunk's buffered box, snapped to the cells, keeps each chunk's raster on
    that grid):

    >>> def canopy(cloud, chunk):                          # doctest: +SKIP
    ...     x0, y0, x1, y1 = chunk.outer
    ...     b = (np.floor(x0), np.floor(y0), x1, y1)
    ...     return ground.make_chm(cloud, 1.0, bounds=b)
    >>> chm = als.apply(cat, canopy, buffer=0)             # doctest: +SKIP

    Numbers per chunk come back as a list:

    >>> counts = als.apply(cat, lambda c, ch: int((~c.attrs["buffer"]).sum()))  # doctest: +SKIP
    """
    cat = _as_catalog(catalog)
    if not callable(fn):
        raise ValueError("fn must be callable as fn(cloud, chunk)")
    fmt = _format(format)
    chunks = cat.chunks(chunk_size, buffer, origin)
    if not chunks:
        return []
    core = cat._core()
    n_workers = _core.als_workers([c.est_points for c in chunks], _workers(workers),
                                  int(_core.ALS_BYTES_PER_POINT))
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)

    def job(i: int):
        ch = chunks[i]
        xyz, attrs, buf = _core.als_read_chunk(core, ch._core())
        if buf.all():
            return None
        attrs[_BUFFER_ATTR] = buf
        res = fn(PointCloud(xyz, attrs), ch)
        if isinstance(res, PointCloud):
            res = _strip_buffer(res, ch)
            if out is not None:
                if len(res) == 0:
                    return None
                like = ch.own if ch.own is not None else ch.files[0]
                ext = fmt or (Path(cat.tiles[like].path).suffix.lower().lstrip(".") or "laz")
                ext = ext if ext in ("las", "laz") else "laz"
                path = _core.als_output_path(core, str(out), ch.name, ext)
                _core.als_write_like(core, like, path, res.xyz, res.attrs)
                return _Written(path)
        return res

    results: list = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=n_workers) as ex, \
            progress.task("processing ALS chunks", len(chunks)) as bar:
        order = iter(range(len(chunks)))
        running = {i: ex.submit(job, i) for i in itertools.islice(order, n_workers)}
        for i in range(len(chunks)):
            try:
                results[i] = running.pop(i).result()
            except BaseException:
                for f in running.values():
                    f.cancel()
                raise
            bar.update()
            nxt = next(order, None)
            if nxt is not None:
                running[nxt] = ex.submit(job, nxt)
    return _assemble(cat, chunks, results, out)


@dataclass(frozen=True)
class _Written:
    path: str


def _assemble(cat: Catalog, chunks: list[Chunk], results: list, out):
    got = [r for r in results if r is not None]
    if got and all(isinstance(r, _Written) for r in got):
        return _written([r.path for r in got], cat)
    if out is not None and not got:
        return Catalog([], tolerance=cat.tolerance)
    if got and all(isinstance(r, PointCloud) for r in got):
        return PointCloud.concatenate(got)
    if got and all(isinstance(r, Raster) for r in got):
        res = float(got[0].resolution)
        if any(abs(float(r.resolution) - res) > 1e-9 * res for r in got):
            raise ValueError("the rasters returned differ in resolution; they cannot be joined")
        parts = [(np.ascontiguousarray(r.data, dtype=np.float64), float(r.xmin), float(r.ymin),
                  float(r.resolution), c.core)
                 for r, c in zip(results, chunks, strict=True) if r is not None]
        return cat._raster(_core.als_mosaic(cat._core(), res, parts))
    return [r.path if isinstance(r, _Written) else r for r in results]


def _run_kw(chunk_size, buffer, workers) -> dict:
    return {"chunk_size": None if chunk_size is None else float(chunk_size),
            "buffer": float(buffer), "workers": _workers(workers)}
