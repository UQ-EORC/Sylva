# Trees and crowns

From a height-normalised plot to a tree table: stems, DBH, segmentation,
height and crowns. The [plot workflow](quickstart.md) shows the short form;
this page covers each step and its settings.

## 1. Ground and heights

```python
from sylva import ground, filters

cloud = filters.voxel_downsample(cloud, 0.01)
cloud = ground.classify_ground_csf(cloud, cloth_resolution=0.5, rigidness=2)
dtm = ground.make_dtm(cloud, resolution=0.5)
cloud = ground.normalize_height(cloud, dtm)          # adds "height"
chm = ground.make_chm(cloud, resolution=0.5)
dtm.to_geotiff("dtm.tif", crs="EPSG:28355")
```

- **Merged multi-scan plots:** use `classify_ground_csf`, the Cloth
  Simulation Filter ([Zhang et al. 2016](../references.md)). Set `rigidness` to 1
  on steep slopes and 3 on flat ground.
- **Single scans in dense forest:** use `classify_ground_pmf`, the
  progressive morphological filter ([Zhang et al. 2003](../references.md)). Occluded
  cells whose lowest return is canopy can hold the cloth up.
- **Comparable DTMs across dates:** pass the same `bounds` every time, so
  the grids line up.
- **Checking the result:** `dtm.hillshade()` shades the grid as if lit from
  one direction (by default from the north-west, 45° up) and returns another
  `Raster`, 0 to 255. Ridges, pits left by a filter that bit too deep and
  the seams between tiles stand out in shaded relief where a colour ramp
  hides them. It is `gdaldem hillshade` without the GDAL.

## 2. Stem detection

```python
from sylva import trees

stems = trees.detect_stems(cloud)          # list of Tree, ids 1..n
stems[0]                                   # Tree(tree_id=1, x=..., y=..., dbh=..., quality=...)
```

1. The 1–5 m band is cut into slices.
2. Each slice is clustered in 2D.
3. Circles are fitted by RANSAC ([Fischler & Bolles 1981](../references.md)), and their
   angular coverage is checked.
4. Circles are linked upward into stems.
5. DBH is read from a linear taper at 1.3 m.

Detection favours recall. The candidates include low branches, shrubs and
duplicates, and the next steps remove them. Settings that matter:

| Setting | Default | Change when |
|---|---|---|
| `min_radius`, `max_radius` | 0.015, 0.75 m | the smallest or largest stems of interest differ |
| `min_coverage` | 0.12 | single scans: about 0.5, since only the near side is seen |
| `min_arc_deg` | 0 | many candidates are shrubs, mounds or foliage fitted from a short arc: 130 or more rejects them, and with them the stems seen from one side only (below) |
| `max_lean_deg` | 25 | strongly leaning stems (mangroves, windthrow) |
| `slice_min`, `slice_max` | 1.0, 5.0 m | buttresses (raise `slice_min`) or tall shrub layers |

`min_arc_deg` asks for an unbroken arc of points around each circle,
measured in 10° bins. On a cloud thinned to 2–3 cm a thin stem has only two
or three points a bin, and bins stay empty by chance, so the arc is carried
across up to two empty bins in a row where the point count explains them
(a densely sampled circle gets no such allowance). It remains a real
filter: on the four reference plots of *Benchmarks > Tree detection*, 130°
keeps 80 %, 56 %, 66 % and 65 % of the reference trees where the default
keeps 98 %, 74 %, 83 % and 71 %, because many stems are not seen that far
round. It pays where spurious candidates outnumber the trees
it costs (the conifer plot: half the spurious trees gone, F1 0.65 → 0.69)
and not elsewhere, so try it against a count you trust before adopting it.

`trees.dbh_profile(cloud, (t.x, t.y))` measures the stem diameter at a
series of heights (a taper curve). `trees.fit_circle` and
`trees.fit_circle_ransac` are the underlying 2D fits.

## 3. Segmentation and pruning

```python
stems, merged_into = trees.merge_branches(cloud, stems)   # optional: drop limbs detected as stems
labels = trees.segment_trees(cloud, stems)                # tree_id per point, -1 unassigned
stems, labels, _ = trees.merge_touching(cloud, labels, stems)   # a stem that is part of another tree joins it
trees.tree_heights(cloud, labels, stems, percentile=99)   # fills height and n_points
stems, labels = trees.prune_trees(stems, labels, min_height=3.0)
cloud = cloud.with_attrs(tree_id=labels.astype("int32"))
```

`merge_branches` drops the candidates whose path to the ground runs through
another candidate's trunk: low limbs and forks that the detector fitted as
stems. It is optional, and what it does depends on the stand. On the four
reference plots it removes a fifth to two fifths of the spurious trees, and
it lets the surviving trees take more of the understorey. Where trees fork
above the ground that is the whole effect. Where stems share a base it
merges real trees too: at Wytham, with its coppice stools, 22 of the 149
trees found without it are lost (F1 0.861 → 0.798), and 5 on each of two
other plots (*Benchmarks > Tree detection*).
The `sylva trees` command does not run it. Leave it out for coppice and
multi-stemmed stands.

`segment_trees` grows every tree from its stem over a kNN graph by least-cost
paths, after raycloudtools' `rayextract trees`
([Lowe and Stepanas 2021](../references.md);
[Devereux et al. 2026](../references.md)), whose cost for bending a path
the edges carry.

- **Cost.** An edge of length `d` costs `d ** power / max(bend_floor, cos)
  ** bend_power`, with `cos` the cosine of the angle between the edge and a
  reference direction. Going on in that direction is cheap and turning
  across or back is dear, at most `1 / bend_floor ** bend_power` times as
  dear (39 times at the defaults), so paths go up through a tree rather
  than across the understorey.
- **`path_weight`** sets the reference direction: the vertical, turned by
  that share towards the direction in which the path arrived. At 0 every
  edge is judged against the vertical alone; at 1 against the path alone,
  as in raycloudtools, so that a path which has turned out along a branch
  travels on along it cheaply. The default, 0.3, keeps most of the
  preference for what stands above the stem, which is what holds a tree
  together where crowns are in pieces (sparse or heavily occluded clouds),
  and lets a branch keep its outer part against a tree standing under it.
- **`bend_power`, `bend_floor`** set how fast and how far the cost rises.
  On plots scanned densely from many positions, where every tree is a
  connected surface from its base to its twigs, a gentler cost
  (`bend_power=2, bend_floor=0.3`: a turn costs at most 11 times) divides
  crowns better; on simulated plots scanned from five positions it costs a
  little. Numbers in *Benchmarks > Tree detection*.
- **Seeds.** Each stem is seeded along its whole bark shell, from the
  ground to half a metre above `seed_height`, so the trunk belongs to its
  tree at no cost instead of being grown into from breast height against
  the bend cost.
- **The ground.** Points within `ground_clearance` (0.1 m, the noise band
  of a terrain model) of the ground stay out of the graph and unlabelled;
  a stem is labelled to within a hand of the ground. With a ground
  classification, `ground="classification"` keeps out what the classifier
  found instead and the cut drops to zero, so the base is labelled to its
  last point above the ground. Mind the classifier's own threshold: CSF at
  its default 0.3 m calls the lowest 30 cm of a trunk ground too.
- **Connectivity.** The graph joins its components wherever two come within
  `max_edge` of each other, so a crown reaches its trunk through sparsely
  sampled branches whatever points the thinning kept. Without this the
  labels of trees of 5 to 10 m depended on which point survived each voxel
  (see *Benchmarks > Tree detection*).
- **`height_prior`** lets tall trees win contested crown points: each
  tree's costs are divided by a height. That height is the vegetation's
  over the stem (the 95th percentile within `height_prior_radius`, 1.5 m),
  since a tree's own is not known before the plot is segmented. A tree
  standing free is costed by its own height; under a closed canopy every
  stem, sapling or dominant, is costed by the canopy's, and the prior
  separates little.
- **`voxel_size`** (3 cm) sets the graph resolution, on a grid anchored at
  the cloud's minimum corner in whole metres. Labels are copied to
  every point. The graph is what segmentation costs in memory: about 60
  bytes a node, plus the node coordinates and the search's own arrays, and
  on a 2 cm cloud 3 cm voxels keep more than half the points as nodes
  (5 cm keep a third, 10 cm an eighth). Read a large plot with
  `sylva.read(path, attrs=["height"])` so that nothing else is held beside
  it (see [Large clouds](io.md#large-clouds)).
- **Understorey sources** (`understorey_height`, off by default). After
  raycloudtools, near-ground points away from every detected stem can be
  sources too, so shrubs and saplings keep their own points instead of
  flowing into the nearest tree. Without them a tree holds a good deal that
  is not the tree: on the four CHERLET reference plots, 26 to 71 % of the
  points below 2 m that belong to no reference tree are given to one.
  `understorey_height=10` brings that down to 9 to 64 %, releasing 2 to 13
  points of understorey for each point it takes from a real tree, and costs
  about two small trees in a hundred on two of the plots, whose crowns the
  sources reach from below. Turn it on when the lower two metres of
  each tree matter (stem models, understorey work); leave it off when the
  tree list does. The `low_height` rule is separate and always applies:
  below 0.5 m a tree keeps only what lies within `low_radius` of its stem.
  Numbers in *Benchmarks > Tree detection*.

`merge_touching` looks at the trees once they are segmented. Detection
favours recall, so some candidates are not trees of their own: a second
circle on a wide trunk, a low fork, a whorl of branches fitted as a stem.
Each is grown like any other stem and comes away holding a piece of its
parent's crown, and that piece lies against the parent along much of its
extent, where two separate trees touch only at the edge of their crowns.
The step thins the labelled points above 0.5 m to 5 cm, counts for each
pair of trees the share of the smaller one's points that have a point of
the larger within 10 cm, and merges the smaller into the larger while that
share is at least `threshold` (0.1), largest share first. Only trees that
reach `min_top` (3 m) take part, so that a shrub against a trunk is
dropped by pruning and not added to the tree. It takes a few seconds on a
plot. Measured as paired runs it removed spurious trees and lost none on
any data set it was tried on; a lower threshold removes more and begins to
merge real neighbours (*Benchmarks > Tree detection*). Unlike
`merge_branches` it left the stems of coppice stools apart. `sylva trees`
runs it; `--merge-touching 0` turns it off.

`prune_trees` removes candidates lower than `min_height` and merges stems
closer than `merge_radius`. In conifer stands with many low branches,
`min_quality_short=0.15` removes the remaining false stems.

`basal_area` sums the stems' cross-sections at breast height, in m²/ha.
Keep only the stems inside the plot so that they and `area` cover the same
ground; `min_dbh` sets an inventory threshold:

```python
radius = 50.0
in_plot = [t for t in stems if np.hypot(t.x, t.y) <= radius]
ba = trees.basal_area(in_plot, np.pi * radius**2, min_dbh=0.1)   # m²/ha
```

Against manually segmented plots (F1 at IoU ≥ 0.5; see
[Benchmarks](../benchmarks/trees.md)):

| Plot | F1 |
|---|---|
| Litchfield savanna | 0.96 |
| Wytham temperate broadleaf | 0.77 |
| Ofental conifer | 0.57–0.63 |
| Robson Creek rainforest | 0.57 |

The losses are crown leakage between interlocking neighbours.

### Correcting labels by hand

Where the segmentation has to be right (a reference set, a permanent plot,
training data), fix it in [Segfix](https://github.com/UQ-EORC/Segfix), a GUI
for reassigning, splitting and dismissing points of a segmented cloud. It
finds a `tree_id` column on its own and patches only the label bytes on save,
so every other field survives.

```python
sylva.write(cloud.with_attrs(tree_id=labels.astype("int32")), "plot_trees.laz")
#   pip install segfix && segfix        -> open plot_trees.laz, fix, save
fixed = sylva.read("plot_trees.laz")
labels = fixed.attrs["tree_id"]
```

Write `tree_id` as `int32`. Segfix writes `-1` for points it has dismissed as
*noise*, so the first time it opens a cloud with `-1` labels it asks whether
they mean unassigned (Sylva's reading) or noise, and remembers the answer
with the project; answer **Unassigned**. Points with no tree come back as
`-1`, so there is nothing to map either way.

## 4. Crowns and the tree table

```python
import pandas as pd

crowns = trees.crown_metrics_all(cloud, labels)            # area, base, depth, diameter
table = pd.DataFrame([{**t.as_dict(), **crowns.get(t.tree_id, {})} for t in stems])
table.to_csv("trees.csv", index=False)

shape = trees.crown_shape(cloud[labels == 1], base_xy=(stems[0].x, stems[0].y))
shape["volume"], shape["asymmetry"], shape["offset_direction"]
```

`crown_metrics` finds the crown base from the vertical point distribution.
`crown_shape` stacks convex hulls every 0.5 m, which follows the crown's
taper better than a single 3D hull, and reports the crown's offset from the
stem.

For tree architecture (branch angles, branch orders, volume), build a QSM
per tree: see [QSMs](qsm.md). `QSM.metrics()` gives the same crown
description from the branches.

## What TLS misses

- **Tree height:** the tops of tall trees are often not seen from the
  ground. Run `voxels.tree_sampling` on a ray-traced grid to flag trees
  whose top was not observed (see [Ray-traced voxels](voxels.md)).
- **Point-based crown areas** are lower bounds where neighbours hide one
  another.
