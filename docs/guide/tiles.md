# Plots in tiles

A large plot does not fit in memory with room to process it. The TERN
Tumbarumba hectare, scanned from 138 positions, is 306 million points at
2 cm; processed as one cloud it needed about 55 GB of a 62 GB machine, and
merging the scans had to be split into strips by hand. `sylva.tiles` keeps
such a plot as square LAS/LAZ tiles and processes one tile at a time, each
with a buffer of points from its neighbours, so that memory is bounded by
the tile size rather than the plot size. The results are those of the
single-cloud functions on the whole plot: the same points kept, the same
values, the same stems.

The tile engine is the one of [Airborne lidar tiles](als.md): a catalogue
known from the file headers, chunks with buffers, results that do not
depend on the number of workers. `tiles.Catalog`, `tiles.catalog`,
`tiles.apply`, `tiles.retile` and `tiles.write_tiles` are the objects of
`sylva.als`, and the `als` functions run on terrestrial tiles unchanged.

## A worked example

Five simulated scans of a 30 m synthetic plot, tiled, cleaned, classified,
normalised and searched for stems:

```python
import numpy as np
from sylva import PointCloud, filters, synthetic, tiles, trees

rng = np.random.default_rng(1)
stems = [(x, y, d, h) for x, y, d, h in zip(rng.uniform(3, 27, 12), rng.uniform(3, 27, 12),
                                             rng.uniform(0.2, 0.5, 12), rng.uniform(12, 20, 12))]
scene = synthetic.forest(stems, size=30.0, ground_points=60_000, margin=2.0)
positions = [(5, 5), (15, 15), (25, 5), (5, 25), (25, 25)]
scans = [synthetic.scan(scene, origin=(x, y, 1.5), resolution_deg=0.25).to_pointcloud()
         for x, y in positions]

cat = tiles.from_scans(scans, "tiles", tile_size=10.0, voxel_size=0.02)
print(cat, tiles.last_run().max_points)        # Catalog(25 tiles, 402,053 points) 171824

clean = tiles.statistical_outlier_removal(cat, "sor", k=6, std_ratio=1.0)
ground = tiles.classify_ground(clean, "ground", method="pmf", buffer=5.0)
heights = tiles.normalize(ground, "heights", dtm_resolution=0.5, buffer=5.0)
found = tiles.detect_stems(heights, buffer=2.0)
print(f"{len(found)} stems; largest DBH {max(t.dbh for t in found):.2f} m")   # 13 stems; 0.49 m
```

Each step equals its whole-cloud counterpart, which the plot is small
enough to check:

```python
whole = PointCloud.concatenate([s.with_attrs(scan_id=np.full(len(s), i, np.uint32))
                                for i, s in enumerate(scans)])
thinned = filters.voxel_downsample(whole, 0.02, origin=(0, 0, 0))
print(len(thinned) == cat.n_points)                                            # True
kept = filters.statistical_outlier_removal(cat.read(), k=6, std_ratio=1.0)
print(len(kept) == clean.n_points)                                             # True
same = trees.detect_stems(heights.read(), cluster_seeds=True)
print([(t.x, t.y, t.dbh) for t in same] == [(t.x, t.y, t.dbh) for t in found])  # True
```

For real scans, give `from_scans` the files and their registration
(`transforms`, one 4x4 matrix per scan such as the SOP), and `bounds` for
the plot and its buffer; one scan is read at a time.

## Building tiles from scans

`tiles.from_scans(scans, out, tile_size, voxel_size, transforms, origin,
bounds)` takes the scans in order, as files or point clouds:

1. Each scan is moved by its transform (as `PointCloud.transform` moves
   it), cropped to `bounds` and thinned to the first point of each voxel it
   has, on one grid for all the scans (a corner at `origin`, by default
   `(0, 0, 0)`, so voxels lie at multiples of `voxel_size`).
2. Those points go, tile by tile, to scratch files in the output directory
   at full precision, with a `scan_id` attribute (the scan's position in
   the list).
3. Each tile is then assembled from its scratch file, keeping the first
   point per voxel in scan order, and written as `<xmin>_<ymin>.laz`.

The tiles are cut along voxel boundaries (the tile size must be a whole
number of voxels), so no voxel is split between two tiles, and together the
tiles hold exactly the points that
`filters.voxel_downsample(concatenation, voxel_size, origin=origin)` keeps
of all the scans concatenated in order, with all their attributes; only the
coordinates are rounded to the tiles' `scale` (1 mm by default). In memory
are one scan, and one tile per worker; the scratch files take about 40
bytes per thinned point of each scan, plus its attributes, until the tiles
are written.

Tiles keep whatever attributes the points carry, `tree_id`, `wood`,
`scan_id`, `height` or any other: the LAS dimensions by name, the rest as
typed extra bytes, and every operation below writes them back unchanged.

## What each operation needs

Every operation runs one chunk per tile: the tile's own file is its core,
the points of the other files within `buffer` metres are read around it,
and the output tile holds the core points only.

| Function | Whole-cloud result it reproduces | Buffer | Exactness |
|---|---|---|---|
| `from_scans` | `filters.voxel_downsample` of the scans concatenated, `origin` given | none | exact |
| `voxel_downsample` | `filters.voxel_downsample(cat.read(), size, origin)` | one voxel (set automatically) | exact |
| `statistical_outlier_removal` | `filters.statistical_outlier_removal` | any (1 m saves second reads) | exact, two passes |
| `radius_outlier_removal` | `filters.radius_outlier_removal` | any (the radius saves second reads) | exact |
| `estimate_normals`, `planarity_linearity` | `filters.estimate_normals`, `filters.planarity_linearity` | any (1 m saves second reads) | exact but for distance ties (see Limitations) |
| `classify_ground(method="pmf")` | `ground.classify_ground_pmf` | see below | exact with the buffer below |
| `classify_ground(method="csf")` | `ground.classify_ground_csf` | 5 to 10 m | close, not exact |
| `dtm` | `ground.make_dtm` | widest gap in the ground | exact away from the plot edge |
| `normalize` | `ground.normalize_height` with `make_dtm` | widest gap in the ground | to 1e-14 m away from the plot edge |
| `detect_stems` | `trees.detect_stems(..., cluster_seeds=True)` | about 2 m | exact with that buffer |

**Neighbourhoods at any buffer.** Outlier removal and the local features
look at each point's `k` nearest neighbours or at a sphere of fixed
radius. A point's neighbourhood found in the tile and its buffer is the
true one when it reaches no farther than the nearest edge of the buffer
(beyond which there may be points not read). For the few points whose
neighbourhood reaches farther, isolated points mostly, the tile is read
again, widened just enough to hold each one's reach, which bounds its true
neighbourhood, and they are evaluated from that. So the result never
depends on the buffer; a buffer wider than almost every neighbourhood only
saves the second reads, which `tiles.last_run()` counts.

**Statistical outlier removal in two passes.** CloudCompare's SOR removes
points whose mean distance to their `k` neighbours exceeds `mean + std_ratio
* std`, and the mean and standard deviation are those of that distance over
the whole cloud. A tile with its buffer has its own mean and spread, so a
filter run tile by tile (as `als.filter(method="sor")` does) keeps
different points. The tiled version computes every point's mean distance
first, tile by tile, into scratch files; then the global mean and standard
deviation, summed in catalogue order exactly as the whole-cloud filter
sums them; then applies the threshold as it writes the tiles.

**Voxel thinning.** Points are taken in catalogue order (tile by tile,
each in file order), so a voxel that straddles two tiles keeps one point,
from the first tile that has one, as it does in the whole cloud. Points of
one voxel are less than a voxel apart, so a buffer of one voxel suffices.

**Ground.** The progressive morphological filter grids the points on
cells at multiples of `cell_size` in every tile, as the whole cloud does,
and each of its openings looks `2 w` cells away (half-width `w` = 1, 2, 4,
... up to `max_window / cell_size`). A buffer of `2 * cell_size * (1 + 2 +
4 + ... + W)` plus the widest gap in the ground therefore gives the
whole-cloud classes everywhere; in practice much less does (5 m on the
synthetic plots of the tests, 10 m on Tumbarumba with the default 10 m
window). The cloth simulation is not exact: the cloth is one sheet, and
its settling and its stopping test depend on all of it, so near a tile
edge a few points can be classified differently (below 0.1 % of the points
on the synthetic plots with a 5 m buffer). `dtm` and `normalize` take each
cell from the tile whose core holds it; with a buffer wider than the
widest gap in the ground they give the whole-cloud DTM cell for cell, and
heights that differ from the whole-cloud ones by rounding only, away from
the outer edge of the plot (where the whole-cloud grid stops at the ground
points and the tiled one at the tiles).

**Stems.** Each tile searches its points and buffer and keeps the stems
whose position (at the reference height) lies in its core, or nearest to
it, so every stem is kept once. The stem detector draws RANSAC samples
from random streams; by default one stream runs through all the clusters
of a layer, so a stem's circles depend on every cluster before it.
`detect_stems` here gives each cluster its own stream (`cluster_seeds=True`,
also an option of `trees.detect_stems`), seeded from its position, so a
stem depends only on the points around it. With a buffer wider than the
clusters and circles of a stem at a tile edge (clusters are at most
`max_cluster_extent`, 2 m, wide) the stems, their order and their ids are
those of `trees.detect_stems(whole, cluster_seeds=True)`. Compared with the
default single stream, the same stems are found, their fits differing
where RANSAC settles on another circle.

## Memory

At most `workers` tiles are in memory at a time, each with its buffer:
about 256 bytes per point, the same budget as for airborne tiles (see
[`sylva.limits`](../api/limits.md)), and fewer workers are used when that
many would not fit. `tiles.last_run()` returns what the last operation
held:

```python
tiles.statistical_outlier_removal(cat, "sor", k=6, std_ratio=1.0)
print(tiles.last_run())
# RunInfo(chunks=25, max_points=..., points_read=..., rereads=..., widened_points=...)
```

`max_points` is the most points held by one tile with its buffer (or by a
widened read, or by `from_scans` for its largest scan or tile). For 10 m
tiles with a 1 m buffer a tile holds 1.44 times its own points; the tests
check on a 60 m plot of 36 tiles that no step holds more than an eighth of
the plot.

## Checked on Tumbarumba

A 30 x 30 m corner of the Tumbarumba plot (TUMBA_2022) was tiled from the
per-scan files and processed tile by tile and as one cloud:

133 registered scans (the per-scan files, 2 cm thinned, with their
registration), cropped to the corner and thinned to 2 cm on one global
grid, in 10 m tiles: 19.6 million points in 15 tiles. The whole-cloud
reference is the tiles read back as one cloud.

| Step | Tiled against whole cloud |
|---|---|
| `from_scans` (24.4 M points in, 19.6 M kept) | the same points as `voxel_downsample` of the concatenated scans; coordinates within the 0.1 mm quantisation |
| SOR, `k=6`, `std_ratio=1` (1 m buffer; 84 points re-evaluated from a wider read) | the same 17,889,417 points kept |
| ROR, 5 cm, 4 neighbours | the same 17,601,527 points kept |
| `voxel_downsample`, 5 cm | the same 6,723,984 points |
| normals, `k=12` (0.5 m buffer) | identical for all but 4,545 points (0.02 %): 3,714 differ by rounding (below 1e-15) and 831 have another neighbour at a tie of the 12th distance (see Limitations) |
| PMF ground (10 m buffer), on the 5 cm cloud | identical classes |
| CSF ground (10 m buffer) | 8,868 of 6.7 M points (0.13 %) classified differently |
| DTM, 0.5 m (5 m buffer) | identical cells, except in the outermost two rows of cells at the plot edge |
| heights (5 m buffer) | within 1e-14 m away from the plot edge (2 m) |
| stems (`min_arc_deg=130`, 2, 5 and 10 m buffers) | the same 18 stems, positions, DBH and ids |

Memory, as the peak resident size of the process: the whole-cloud steps on
the 19.6 M points peaked at 3.2 GB (and 4.5 GB to hold and thin the
concatenated scans); tiled with one worker, every step stayed under 1.0 GB
(`max_points` at most 5.6 M, a tile with its buffer), and building the tiles
peaked at 2.0 GB with eight workers, set by the largest scan. With eight
workers the tiled steps peaked at 4.3 GB and ran in 100 s against 37 s for
the whole-cloud steps: tiling pays in reading and writing files, and in
memory it scales with the tile and the workers, not with the plot, which on
the full hectare is what keeps the run within the machine.

## Limitations

- Exactness is to the last bit for the kept points and the per-point values,
  with two caveats. The nearest-neighbour search orders neighbours at equal
  distances by the layout of its search tree, which differs between a tile
  and the whole cloud. Quantised coordinates make such ties common: where
  one falls at the `k`-th distance, a feature can be computed from another
  neighbour, and where it falls earlier, the covariance is summed in
  another order (a difference below 1e-15). Normals and planarity are
  affected (0.02 % of the Tumbarumba points); outlier removal is not, as it
  uses only the distances. Coordinates are quantised to the tiles' scale,
  so the whole cloud to compare with is the tiles read back (`cat.read()`),
  not the scans before tiling.
- The cloth simulation filter is close to, but not exactly, the whole-cloud
  classification.
- Stem detection in coregistration mode (`cluster_grid_at_slice_min`) grids
  each layer from its own minimum, which depends on the tile; the default
  absolute grid is needed for tiled detection to match.
- A widened second read can be large for a point far from everything else
  (an isolated return metres from the plot); it is refused if it would not
  fit in the memory budget. Removing such points first (radius outlier
  removal needs no second read with its default buffer) avoids it.
- Scratch files of `from_scans` and of statistical outlier removal need
  disk space beside the output.
