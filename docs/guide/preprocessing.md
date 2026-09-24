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

## Coregistering a survey

`sylva.coreg` registers every scan position of a survey at once, with no
targets and no starting alignment: it matches the stems each scan sees. It is
a port of tlsalign. Where scans share RiSCAN targets, it uses them too.

```python
from sylva import coreg, read_riscan_project, riscan

project = read_riscan_project("survey.PROJ")
scans = project.with_scans(require_sop=False)
config = coreg.CoregConfig(
    riscan_filter="current",                              # or "legacy", "none"
    riegl_options=coreg.reading_options("export.settings", min_reflectance=-20),
)
result = coreg.coregister(
    [p.rxp for p in scans], config,
    names=[p.name for p in scans],
    levelling=[p.levelling for p in scans],               # tilted scans levelled first
    reflectors=[p.reflectors() for p in scans],           # RiSCAN targets, if any
    approximate_positions=riscan.gnss_to_local([p.gnss for p in scans]),
)
print(result.report())
result.save("transforms.json")
merged = coreg.merge_clouds([p.rxp for p in scans], result, voxel=0.02)
```

It works in three stages:

1. **Each scan** gets a terrain model and a stem map. Only the locally
   planar points (stems, ground, logs) are kept for ICP.
2. **Each pair** is matched on shared targets where both scans saw three or
   more. Otherwise the two stem maps are matched. A robust point-to-plane ICP
   then refines the match. A pair is accepted only if it fits both overall
   and above the ground, so a flat ground can't confirm a wrong match on its
   own.
3. **The whole survey** is solved as a pose graph, with outlier edges
   rejected. Scans that are left over are retried against the combined
   survey.

Options:

- **Trusted targets.** A pair that shares at least
  `trusted_reflector_matches` (5) targets, with an RMSE under
  `trusted_reflector_rmse` (3 cm), keeps the target solution instead of ICP.
- **Fixed poses and priors.** `fixed={index: pose}` holds scans at trusted
  poses, so new positions can join an existing project. `priors` (RiSCAN
  SOPs, GNSS and compass) place scans that see too few stems to match (see
  `coreg.place_from_prior`).
- **RiSCAN filters.** `riegl_options` applies RiSCAN export bounds on range,
  deviation, reflectance and amplitude, read from an export settings file
  (explicit bounds override the file). `riscan_filter` drops what RiSCAN
  drops: `"current"` removes echoes closer than 0.5 m, and `"legacy"` also
  removes isolated weak echoes, as older RiSCAN versions did.
- **Checking the result.** `result.report()` lists every pair. For each one it
  gives how far apart the same trees land from the two scans, a check that
  needs no ground truth.

A survey where the positions share few trees may register only in part from
stems. On a 14-scan VZ-400 survey, stems alone placed 4 scans; with the
RiSCAN targets, all 14 landed within 7 cm of the target-based SOPs. The same
pipeline is the `sylva coreg` command (see [Command line](cli.md)).
