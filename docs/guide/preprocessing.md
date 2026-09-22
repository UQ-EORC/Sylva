# Filtering and registration

## Thinning

TLS density falls with the square of the range, so near-scanner areas
dominate anything that counts points. Thin before processing:

| Function | Keeps | Use for |
|---|---|---|
| `filters.voxel_downsample(c, 0.01)` | first point per voxel, attributes kept | general thinning, fastest |
| `filters.voxel_downsample(c, 0.01, "centroid")` | voxel centroids, no attributes | smoothing noise |
| `filters.min_distance_subsample(c, 0.01)` | points at least `d` apart, no grid pattern | normals, segmentation |
| `filters.random_subsample(c, fraction=0.1, seed=0)` | random share, reproducible | quick looks, tests |

Typical spacings: 1 cm for stems and QSMs, 2–5 cm for ground filtering and
ICP, and 5–10 cm for plot overviews.

## Cropping and noise

```python
from sylva import filters

plot = filters.crop_cylinder(cloud, center_xy=(0, 0), radius=30)
box = filters.crop_box(cloud, (None, None, 0.5), (None, None, 40))   # None = open side
near = filters.range_filter(scan, origin=(0, 0, 0), max_range=60)
clean = filters.statistical_outlier_removal(cloud, k=8, std_ratio=2.0)
clean = filters.radius_outlier_removal(cloud, radius=0.05, min_neighbors=4)
keep = filters.statistical_outlier_removal(cloud, return_mask=True)      # boolean mask instead
```

The statistical filter compares every point against global statistics.
Where density varies a lot, run it per scan or after thinning, or it removes
distant, sparse points that are real.

## Local geometry

```python
normals = filters.estimate_normals(cloud, k=12)            # sign arbitrary
planarity, linearity = filters.planarity_linearity(cloud, k=20)
labels = filters.euclidean_clusters(cloud.xyz, radius=0.1, min_points=50)   # 0 = largest, -1 = too small
dist, idx = filters.knn(cloud.xyz, queries, k=8)
```

Stems and branches are linear, while leaves and ground are planar. These
features drive the leaf/wood filters in `sylva.qsm` and `sylva.leaves`.

## Registration

Scans from a RiSCAN project already carry their SOPs (see
[Point clouds and files](io.md)). To register scans yourself, or to refine
a registration:

```python
from sylva import registration as reg, filters

# 1. Coarse: from matched targets (reflectors, tie points)
T0 = reg.kabsch(source_targets, target_targets)

# 2. Fine: ICP on thinned clouds, starting from T0
src = filters.voxel_downsample(scan_b, 0.05)
dst = filters.voxel_downsample(scan_a, 0.05)
T, info = reg.icp(src, dst, init=T0, max_correspondence_distance=0.3,
                  method="plane", trim=0.8)
info                    # {'rmse': ..., 'iterations': ..., 'n_correspondences': ...}

merged = reg.merge_scans([scan_a, scan_b], [np.eye(4), T])   # adds scan_id
```

- **Starting point.** ICP only converges from a start within about
  `max_correspondence_distance`, so give it the SOP or a Kabsch estimate.
  `reg.rotation_z` and `reg.translation` build simple starting guesses.
- **Partial overlap.** `trim` below 1 drops the worst pairs each iteration,
  which is what two scan positions of a forest plot need.
- **Checking the result.** ICP's RMSE mixes noise with misregistration.
  `quality.stem_noise` separates the two, giving the horizontal offset of
  every scan measured on the stems (see [Scan quality](quality.md)).
