# Plot workflow

From a plot point cloud to a stem map, segmented trees and a density profile.
To follow along on real data, use the Litchfield tile the
[example notebooks](../examples/index.md) run on
(`docs/examples/data/litch_tile.laz`).

```python
import sylva
from sylva import ground, trees, canopy

cloud = sylva.read("plot.laz")
cloud = ground.classify_ground_csf(cloud)          # adds classification (2 = ground)
dtm = ground.make_dtm(cloud, resolution=0.5)
cloud = ground.normalize_height(cloud, dtm)        # adds height attribute

stems = trees.detect_stems(cloud)                  # DBH from 1.3 m slices
labels = trees.segment_trees(cloud, stems)         # tree_id per point
trees.tree_heights(cloud, labels, stems)
stems, labels = trees.prune_trees(stems, labels)   # drop understorey clumps, merge duplicates
crowns = trees.crown_metrics_all(cloud, labels)
for t in stems:
    print(t.tree_id, f"DBH {t.dbh:.3f} m, height {t.height:.1f} m,",
          f"crown {crowns[t.tree_id]['crown_area']:.1f} m2")

z, pad = canopy.pad_profile_voxel(cloud, voxel_size=0.5)
sylva.write(cloud.with_attrs(tree_id=labels), "plot_segmented.laz")
```

!!! tip "Ground filter for single scans"
    For single-scan TLS in dense forest prefer `classify_ground_pmf`: the
    morphological opening removes occluded cells whose lowest return is canopy,
    which trip up the cloth simulation (`classify_ground_csf` suits multi-scan
    plots and open terrain).
