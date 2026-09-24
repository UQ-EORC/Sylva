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
| `min_arc_deg` | 0 | merged plots: 130 or more rejects arcs that are too short |
| `max_lean_deg` | 25 | strongly leaning stems (mangroves, windthrow) |
| `slice_min`, `slice_max` | 1.0, 5.0 m | buttresses (raise `slice_min`) or tall shrub layers |

`trees.dbh_profile(cloud, (t.x, t.y))` measures the stem diameter at a
series of heights (a taper curve). `trees.fit_circle` and
`trees.fit_circle_ransac` are the underlying 2D fits.

## 3. Segmentation and pruning

```python
stems, merged_into = trees.merge_branches(cloud, stems)   # drop limbs detected as stems
labels = trees.segment_trees(cloud, stems)                # tree_id per point, -1 unassigned
trees.tree_heights(cloud, labels, stems, percentile=99)   # fills height and n_points
stems, labels = trees.prune_trees(stems, labels, min_height=3.0)
cloud = cloud.with_attrs(tree_id=labels.astype("int32"))
```

`segment_trees` grows every tree from its stem over a kNN graph by least-cost
paths, after raycloudtools' `rayextract trees`
([Devereux et al. 2026](../references.md)).

- **Cost.** Climbing is cheap, while horizontal and downward steps are
  expensive, so paths go up through a tree rather than across the
  understorey.
- **`height_prior`** lets tall trees win contested crown points.
- **`wood_costs=True`** stops foliage from bridging neighbouring crowns. It is
  slower.
- **`voxel_size`** (3 cm) sets the graph resolution. Labels are copied to
  every point.
- **Understorey competes** (`understorey_height`, on by default). After
  raycloudtools, near-ground points away from every detected stem are
  sources too, so grass, shrubs and saplings keep their own points instead
  of flowing into the nearest tree. That matters most for QSMs: before, the
  understorey around a stem was modelled as a fan of low branches. On the
  CHERLET Litchfield test block it raises F1 from 0.73 to 0.83 and cuts
  the share of tree points that are really understorey from 17 % to 6 %.
  Wytham and Robson Creek improve too.

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
training data), fix it in [Segfix](https://github.com/tim-devereux/segfix), a
GUI for reassigning, splitting and dismissing points of a segmented cloud. It
finds a `tree_id` column on its own and patches only the label bytes on save,
so every other field survives.

```python
ids = np.where(labels > 0, labels, 0).astype("int32")   # Segfix: 0 = unassigned
sylva.write(cloud.with_attrs(tree_id=ids), "plot_trees.laz")
#   pip install segfix && segfix        -> open plot_trees.laz, fix, save
fixed = sylva.read("plot_trees.laz")
labels = np.where(fixed.attrs["tree_id"] > 0, fixed.attrs["tree_id"], -1)
```

Map Sylva's `-1` (unassigned) to `0` first and use `int32`: Segfix reads a
negative id as its own *noise* marker.

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
