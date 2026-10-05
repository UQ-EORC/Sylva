# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Large-area airborne lidar (ALS and UAV lidar): tiles, chunks and buffers.

An airborne survey comes as hundreds of LAS/LAZ tiles, together far larger
than memory, and no tile can be processed on its own: a ground filter or a
DTM cell at a tile edge needs the points across it. This module follows
lidR's ``LAScatalog`` (Roussel et al. 2020):

1. :func:`catalog` reads only the file headers (extent, point count, point
   format, CRS) and checks the tiling for overlaps, holes, mixed CRS and
   mixed formats, without reading a point.
2. The area is divided into chunks, one per tile or a regular grid
   (``chunk_size``), each read with a ``buffer`` of points from its
   neighbours. Only the files that overlap a chunk are opened, and only the
   points inside its buffered box are kept.
3. Chunks run in parallel, as many at a time as the memory budget of
   :mod:`sylva.util.limits` allows, and results are assembled in chunk order, so
   they do not depend on the number of workers. Point outputs keep only the
   chunk's own (core) points; raster outputs are joined into one seamless
   raster on a grid shared by the whole catalogue.

:func:`apply` runs any function this way; :func:`classify_ground`,
:func:`dtm`, :func:`chm`, :func:`normalize`, :func:`filter`,
:func:`retile` and :func:`decimate` are built on the same engine, in the
Rust core, as are the area-based metrics of :mod:`sylva.als.metrics`
(:func:`grid_metrics`, :func:`plot_metrics`), available here too.
:func:`sylva.synthetic.als_flight` simulates an airborne survey to try
them on.

Examples
--------
>>> from sylva import als
>>> cat = als.catalog("tiles/")                            # doctest: +SKIP
>>> print(cat.report())                                    # doctest: +SKIP
>>> ground = als.classify_ground(cat, "ground/")           # doctest: +SKIP
>>> dtm = als.dtm(ground, resolution=1.0)                  # doctest: +SKIP
>>> chm = als.chm(ground, resolution=0.5)                  # doctest: +SKIP
"""

from __future__ import annotations

from .catalogue import Catalog, Chunk, Tile, catalog  # noqa: F401
from .engine import apply  # noqa: F401
from .ops import chm, classify_ground, decimate, dtm, filter, normalize, retile, write_tiles  # noqa: F401

__all__ = [
    "Tile", "Catalog", "Chunk", "catalog", "apply", "classify_ground", "dtm", "chm", "normalize",
    "filter", "retile", "decimate", "write_tiles", "grid_metrics", "pixel_metrics", "plot_metrics",
    "cloud_metrics", "metric_names", "PlotMetrics",
    # Ray-based canopy structure (sylva.als.canopy)
    "Trajectory", "read_trajectory", "estimate_trajectory", "pulses", "ALSProfile", "gap_profile",
    "ALSVoxels", "ray_voxelize", "week_seconds",
]


# The area-based metrics live in their own module and build on this one.
from .metrics import (  # noqa: E402
    PlotMetrics,
    cloud_metrics,
    grid_metrics,
    metric_names,
    pixel_metrics,
    plot_metrics,
)

# Ray-based canopy structure lives in its own module; it uses this one's engine.
from .canopy import ALSProfile, ALSVoxels, gap_profile, ray_voxelize  # noqa: E402
from .trajectory import Trajectory, estimate_trajectory, pulses, read_trajectory, week_seconds  # noqa: E402

# Individual trees (tree tops, crowns, labelled tiles) live in their own module.
from .trees import (  # noqa: E402
    LinearWindow,
    Trees,
    TreeTops,
    crown_hull,
    find_trees,
    li2012,
    locate_trees,
    segment_crowns,
    segment_trees,
)

__all__ += ["LinearWindow", "TreeTops", "Trees", "locate_trees", "segment_crowns", "li2012",
            "crown_hull", "segment_trees", "find_trees"]

# Submodules that build on the names above.
from . import canopy, catalogue, engine, metrics, ops, tiles, tiles_trees, trajectory, trees  # noqa: E402,F401,F811
