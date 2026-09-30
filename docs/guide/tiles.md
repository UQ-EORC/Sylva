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
neighbourhood reaches farther, isolated points mostly, the tiles within
each one's reach (which bounds its true neighbourhood) are read again one at
a time, each point keeping its nearest candidates over them, and they are
evaluated from those. So the result never depends on the buffer, and memory
stays at one tile even for a return far above the canopy whose neighbours
lie across the plot; a buffer wider than almost every neighbourhood only
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

## Trees

`tiles.segment_trees(catalog, stems, out)` runs the segmentation sequence
of a whole plot, `trees.merge_branches`, `trees.segment_trees`,
`trees.tree_heights` and `trees.prune_trees`, over height-normalised tiles,
and writes the tiles again with each point's tree id in `tree_id` (int32,
-1 for none). Continuing the example above:

```python
trees_, seg = tiles.segment_trees(heights, found, "segmented", buffer=10.0,
                                  merge={"voxel_size": 0.1}, prune={"min_height": 2.0},
                                  percentile=99.0, voxel_size=0.1)
print(len(trees_), tiles.last_run().max_points)          # 12 108292

kept, _ = trees.merge_branches(whole_h, found, voxel_size=0.1, voxel_origin=(0, 0, 0))
labels = trees.segment_trees(whole_h, kept, voxel_size=0.1, voxel_origin=(0, 0, 0))
trees.tree_heights(whole_h, labels, kept, percentile=99.0)
same, labels = trees.prune_trees(kept, labels, min_height=2.0)
print([(t.tree_id, t.height) for t in same] == [(t.tree_id, t.height) for t in trees_])  # True
```

(`whole_h` is `heights.read()`.) How it works:

1. **Graph nodes.** The segmentation builds its graph on the points above
   `cut_above_ground`, thinned to the first point of each `voxel_size`
   voxel. Tiled, the voxels are those of one grid with a corner at
   `voxel_origin` (`(0, 0, 0)` by default), and the first point of each is
   found tile by tile in catalogue order, as the tiled voxel thinning finds
   it; these are the nodes the whole cloud has, given the same corner
   (`trees.segment_trees(..., voxel_origin=(0, 0, 0))`; without it, the
   whole-cloud grid starts at the cloud's lowest point). The nodes, a small
   share of the points at 0.1 m, are kept in scratch files.
2. **Branches.** Each stem is traced to the ground by the tile whose core
   holds it, on that tile's nodes and those of its buffer; the chains
   (a branch of a branch, mutual pairs) are then resolved over all stems at
   once, as `merge_branches` resolves them.
3. **Paths.** Each tile's nodes and those within `buffer` are segmented
   with every stem that could reach them (stems up to their seed and
   height-prior reach beyond the buffer), and the tile keeps the labels of
   the trees whose stem it holds. A tree's labels are then those of the
   whole plot when its competitors have their stems, seeds and crowns
   within the buffer: a buffer wider than the largest crown. A tile whose
   trees' nodes come within `edge_margin` (2 m) of the edge of its buffer,
   where the plot goes on, is segmented again with a buffer twice as wide,
   up to `max_buffer`; trees still at the edge are named in a warning and
   flagged `crown_at_edge` in `Tree.extra`.
4. **Points.** Every point takes the label of its nearest node, tile by
   tile. The nearest node of a point is its own voxel's or closer, so the
   nodes within one voxel diagonal of the tile hold it.
5. **Trees.** Heights and point counts are gathered per tree over the
   tiles, pruning runs on the tree list (it needs nothing else), and the
   tiles are written with the final ids, unique over the plot.

`merge` and `prune` take `True` (their defaults), `False` (skip) or a dict
of their keywords, so the graph of `merge_branches` keeps its own settings
(`k=10`, `power=3`, ...). The remaining keywords are those of
`trees.segment_trees`.

## One tree at a time

`tiles.split_trees(segmented, "trees")` writes each tree's points, from
every tile it touches and in the order the whole plot has them, to a
`TreeStore`: a directory per tree, at full precision and with every
attribute, and an index of the trees (points, bounds, tiles). One tile per
worker is held. A tree is then read without the plot:

```python
store = tiles.split_trees(seg, "trees")
one = store.read(store.ids[0])             # or tiles.read_tree(store, tree_id)
print(len(store), store.n_points(store.ids[0]), store.tiles_of(store.ids[0]))
# 12 31144 ['0_0.laz', '10_0.laz', '0_10.laz', '10_10.laz']
```

`tiles.read_tree(catalog, tree_id, bounds=...)` reads a tree from the tiles
themselves instead, reading every tile that meets `bounds`.

The per-tree steps of a plot run from the store, a few trees at a time on
threads (Sylva's computations release the interpreter, and each is
parallel inside), largest first, with at most `workers` trees in memory and
fewer when that many of the largest would not fit in the budget
(`BYTES_PER_TREE_POINT`, 1 kB, per point):

| Function | Whole-plot result it reproduces | Written |
|---|---|---|
| `tiles.classify_leaf_wood(store, seg, out)` | `leaves.classify_leaf_wood(plot[labels == t])` for every tree of 100 points or more | `wood` (int8: 1 wood, 0 leaf, -1 none) in the tiles, and per tree in the store |
| `tiles.build_qsms(store, trees_, **options)` | `qsm.build_plot(plot, labels, trees_, **options)`, every option (DBH anchor, `stem_radius_cap`, buttresses) | a `PlotQSMs`; each tree's result is also kept in the store |
| `tiles.crown_metrics(store)` | `trees.crown_metrics_all(plot, labels)` | a dict per tree |
| `store.map(fn)` | `fn(tree_id, plot[labels == tree_id])` | whatever `fn` returns |

Each gives exactly the whole-plot values (the tests compare the leaf / wood
labels, the cylinders of every model and the crown metrics bit for bit),
since each tree's points, in the same order, are all these functions see.
`resume=True` keeps what the store already holds, so an interrupted run
picks up at the next tree.

## The plot workflow

`tiles.run_plot(scans, out, ...)` runs everything from registered scans to
trees and QSMs, and the `sylva tiles-plot` command runs it from the shell:

```python
import numpy as np
from sylva import tiles

st = np.load("coreg_state.npz")                 # corrections from coregistration
scans = [f"scans/{i:03d}.laz" for i in range(len(st["corr"]))]
run = tiles.run_plot(scans, "run", transforms=list(st["corr"]), use=st["use"],
                     bounds=(-10, -10, 110, 110), plot=(0, 0, 100, 100),
                     voxel_size=0.02, sor={"k": 6, "std_ratio": 1.0},
                     ground={"method": "csf", "cloth_resolution": 0.5, "rigidness": 2},
                     stems={"min_arc_deg": 130}, prune={"min_height": 2, "min_slenderness": 10},
                     qsm_options={"stem_radius_cap": 1.5}, workers=4)
```

```
sylva tiles-plot scans/*.laz run --transforms corr.npy --use use.npy \
    --bounds -10 -10 110 110 --plot 0 0 100 100 --min-slenderness 10 --workers 4
```

The stages, each in its own directory of `out` with a `.done` marker
written when it is complete:

| Stage | Output |
|---|---|
| `scans/` | for a RiSCAN project only (a directory or `RiscanProject`): each scan read with its SOP (`read_options`), cropped to `bounds` and thinned; `transforms` are then the corrections applied after the SOPs |
| `tiles/` | `from_scans` with `transforms` (and scans flagged False in `use` left out), `bounds`, `tile_size`, `voxel_size` |
| `sor/` | `statistical_outlier_removal` (`sor`; False skips it) |
| `ground_thin/`, `ground/`, `dtm.asc` | ground classified (`ground`) on a `ground_voxel` (5 cm) thinning, and the DTM (`dtm_resolution`) |
| `heights/` | `normalize` with that DTM |
| `stems.pkl` | `detect_stems` on the tiles inside `plot` (`stems`) |
| `segmented/`, `trees.pkl` | `segment_trees` (`merge`, `segment`, `prune`, `percentile`, `buffer`) |
| `trees/` | `split_trees`, with each tree's leaf / wood and QSM as they are made |
| `wood/` | `classify_leaf_wood` (`leaf_wood`; False skips it) |
| `qsm_table.csv`, `qsm_models.pkl` | `build_qsms` (`qsm_options`; `qsm_files` also writes cylinders and meshes) |
| `trees.csv` | each tree with its crown metrics and QSM volume |

A second call with the same `out` takes every complete stage as it is, and
the leaf / wood and QSM stages resume tree by tree, so an interrupted run
(killed, out of time) continues where it stopped; the tests interrupt a run
while it fits QSMs and check that the rerun does nothing twice and ends
with the tables of an uninterrupted run. A stage's directory is emptied
when it starts, so a stage cut short is done again from its beginning.
Delete a stage's directory (and those after it) to redo it with other
settings. The returned `PlotRun` holds the trees, the table, the models,
the catalogue of every stage, the DTM, the store, and each stage's time and
largest `max_points`.

Memory is set by the tile size, the buffers and `workers`: each tiled stage
holds `workers` tiles with their buffers, the segmentation `workers` tiles'
graph nodes with a buffer of `buffer` (up to `max_buffer`), and the
per-tree stages `workers` trees.

## Voxels after the workflow

Ray-traced voxels are a separate step for now (the voxel traversal is
being made to run in blocks). They need the pulses of the registered scans
and the DTM of the run:

```python
import numpy as np
from sylva import voxels
from sylva.raster import Raster

dtm = Raster.from_ascii_grid("run/dtm.asc")
grid = voxels.ray_voxelize("rays_registered.parquet", 0.5,
                           ((0, 0, np.nanmin(dtm.data) - 1), (100, 100, np.nanmax(dtm.data) + 60)),
                           dtm=dtm, occlusion=True, beam=(0.007, 0.00027))
seg = run.catalogs["segmented"].read()
samp = voxels.tree_sampling(grid, seg, seg.attrs["tree_id"])
```

`rays_registered.parquet` is the scans' pulses (`Shots`) moved by the same
corrections as the points (origins by the matrix, directions by its
rotation). `voxels.tree_sampling` needs the segmented points with their
labels in one cloud; on a plot too large for that, sample the trees one at
a time from the store.

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
tile read again for far-reaching points, by `from_scans` for its largest
scan or tile, or by `segment_trees` for a tile's graph nodes with its
buffer). For 10 m
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

## The whole hectare

The full Tumbarumba hectare was run with `tiles.run_plot` from the 133
registered per-scan files and their saved corrections, with the settings of
the whole-cloud run it is compared with: 2 cm, SOR `k=6`, `std_ratio=1`, CSF
ground on a 5 cm thinning with a 0.5 m DTM, stems with `min_arc_deg=130`,
merge and segmentation graphs at 0.1 m, heights at the 99th percentile,
pruning with `min_height=2`, `min_slenderness=10`, and QSMs with
`stem_radius_cap=1.5`; tiles of 10 m over the plot and a 10 m margin
(x, y from -10 to 110 m), trees from the tiles of the plot (0 to 100 m), a
20 m segmentation buffer and four workers.

**Tiled against whole cloud, same input.** On a 40 x 40 m block of the
plot's height-normalised tiles (16 tiles, 41.8 million points), small
enough to process as one cloud, `segment_trees` (20 m buffer) and the
whole-cloud sequence with the same grid corner gave the same 62 trees,
identical to the bit, and the same label on all 41,786,521 points; each
tree read from the store was the whole cloud's points for it, and the QSMs
built from the store were identical to `qsm.build_plot`'s. The block took
194 s tiled against 58 s as one cloud: the tiled segmentation reads each
tile's buffer, 25 tiles' worth of graph nodes per tile.

**The hectare, against the earlier whole-cloud run.** That run merged the
scans in strips with their own voxel grids, found stems with the default
single random stream, classified ground with CSF on the whole cloud and
cropped the plot before segmenting; the tiled run thins on one global grid,
seeds each stem cluster on its own (`cluster_seeds`), runs CSF with a 10 m
buffer and uses `voxel_origin=(0, 0, 0)`. Each of these moves some points
and some stem fits, so the two runs are close but not equal:

| | Whole cloud | Tiled |
|---|---|---|
| points after SOR | 282,079,298 of 305,345,548 | 281,336,086 of 304,487,376 |
| stem candidates | 645 | 644 |
| trees after pruning | 404 | 410 |
| trees matched within 0.5 m | 345 | 345 |
| basal area (m² ha⁻¹) | 49.25 | 47.23 |
| wood share of tree points | 59 % | 59 % |
| QSMs, total volume | 400 trees, 765.1 m³ | 406 trees, 788.5 m³ |
| peak memory | about 55 GB | 7.4 GB |

For the 345 matched trees the stem positions differ by 6 mm (median), DBH
by 3.3 mm (median; 2.1 % relative, 17.5 % at the 90th percentile, where
RANSAC settled on another circle), height by 1.6 cm (median, 0.1 %; 4.7 %
at the 90th percentile), point counts by 3.1 % (median) and crown area by
1 % (median). QSM volumes of the 342 matched trees modelled in both total
703.3 m³ as one cloud and 735.1 m³ tiled (+4.5 %), with a median difference
per tree of 8.9 %: the models are mostly taper and pipe-model priors (3 %
of the median model's length was fitted to points in both runs), so a
small change in a tree's DBH or points moves its volume. The unmatched
trees (59 whole-cloud only, 65 tiled only; median DBH 0.25 and 0.23 m,
median height 11 and 10 m) are small stems and understorey candidates near
the pruning thresholds that one run's stem fits kept and the other's did
not.

**Memory and time.** No stage held more than 7.4 GB (the resident size of
the process, measured every 5 s): building the tiles 6.8 GB, SOR 1.7 GB,
ground 2.2 GB, heights 1.2 GB, stems 2.4 GB, segmentation 6.9 GB, the tree
store 3.2 GB, leaf / wood 4.0 GB, QSMs 6.5 GB. With four workers the run
took 1 h 41 min (tiles 270 s, SOR 1696 s, ground 803 s, heights 299 s,
stems 307 s, segmentation 1595 s, tree store 287 s, leaf / wood 432 s,
QSMs 336 s, table 14 s), on a machine shared with other jobs (load 13 to 26
on 8 cores); the whole-cloud run took 29 min to the same point, starting
from its merged cloud already on disk. The tiled
steps read and write every tile at each stage and read buffers around each
tile; the per-tree steps are faster tiled, since trees run in parallel.

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
- A point far from everything else (an isolated return tens of metres above
  the canopy) is searched again in every tile within its reach, one tile at
  a time: slow for many such points, but bounded in memory.
- Scratch files of `from_scans`, statistical outlier removal and
  `segment_trees` need disk space beside the output, and a tree store holds
  a second copy of every tree's points (full precision, about 60 bytes a
  point with the usual attributes).
- `segment_trees` matches the whole cloud only on the grid it uses: the
  whole-cloud `trees.segment_trees` anchors its voxel grid at the cloud's
  lowest point unless given `voxel_origin`. Its labels also depend on the
  buffer being wider than the largest crown; a crown that reaches the
  buffer's edge is read again wider, and reported when `max_buffer` is not
  enough. Nearest-neighbour ties (the graph's neighbours, a point's nearest
  node) can, rarely, go another way than in the whole cloud; none did on
  the 41.8 million points of the Tumbarumba block.
- Tiled segmentation is slower than on one cloud (three to four times on
  Tumbarumba), since each tile reads the graph nodes of its whole buffer.
