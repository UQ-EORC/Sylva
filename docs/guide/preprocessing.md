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

The statistical filter ([Rusu et al. 2008](../references.md)) compares every
point's mean distance to its neighbours against global statistics.
Where density varies a lot, run it per scan or after thinning, or it removes
distant, sparse points that are real.

## Local geometry

```python
normals = filters.estimate_normals(cloud, k=12)            # sign arbitrary
planarity, linearity = filters.planarity_linearity(cloud, k=20)
labels = filters.euclidean_clusters(cloud.xyz, radius=0.1, min_points=50)   # 0 = largest, -1 = too small
dist, idx = filters.knn(cloud.xyz, queries, k=8)
```

Planarity and linearity are the eigenvalue features of [Weinmann et al.
(2015)](../references.md). Stems and branches are linear, while leaves and
ground are planar. These features drive the leaf/wood filters in `sylva.qsm`
and `sylva.leaves`.

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

`kabsch` is the closed-form least-squares rotation of Kabsch (1976) with
Umeyama's (1991) guard against reflections. `icp` is point-to-point ICP
(Besl & McKay 1992) or, with `method="plane"`, point-to-plane ICP (Chen &
Medioni 1992) solved by Low's (2004) linearisation; see the
[references](../references.md).

- **Starting point.** ICP only converges from a start within about
  `max_correspondence_distance`, so give it the SOP or a Kabsch estimate.
  `reg.rotation_z` and `reg.translation` build simple starting guesses.
- **Partial overlap.** `trim` below 1 drops the worst pairs each iteration
  (trimmed ICP, Chetverikov et al. 2002), which is what two scan positions
  of a forest plot need.
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
   planar points (stems, ground, logs) are kept for ICP. A scanner tilted
   on its side cannot see the ground in the directions along its tilt axis,
   where the lowest returns are foliage; the terrain there is filled from
   the directions it could see (`ground_min_coverage`).
2. **Each pair** is matched on shared targets where both scans saw three or
   more. Otherwise the two stem maps are matched in plan, and the height
   comes from the ground both scans saw rather than from the stems: a stem's
   height rests on its own scan's terrain model, which is least certain
   under understory, and vertical error is the weak point of stem-based
   registration (Tremblay & Béland 2018; Wang et al. 2023). A robust
   point-to-plane ICP (Chen & Medioni 1992, with Huber weights and a trimmed
   tail as in Chetverikov et al. 2002) then refines the match. A pair is
   accepted only if it fits both overall and above the ground, so a flat
   ground can't confirm a wrong match on its own, and if the two terrain
   models then agree within `max_ground_disagreement` (25 cm).
3. **The whole survey** is solved as a pose graph (Lu & Milios 1997) by
   Levenberg–Marquardt with a Huber (1964) kernel, with outlier edges
   rejected. Each edge is weighted by the directions its ICP surfaces
   constrain: a pair that overlaps mostly on flat ground holds the height
   firmly and the horizontal position loosely. Scans that are left over
   are placed against the combined survey, then registered pairwise to
   their new neighbours.

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
  gives how far apart the same trees land from the two scans (horizontally)
  and how far apart their terrain models are (`dz`), checks that need no
  ground truth.

Positions far apart share little surface, so ICP's fitness falls with the
distance between them; the gates (`min_icp_fitness` 0.04,
`min_icp_fitness_above_ground` 0.03) were set on a 14-scan VZ-400 survey of
upright and tilted scans 30-35 m apart, scored against its reflector-based
registration. From stems alone all 14 scans registered, with a median
error of 2.9 cm and at most 4.4 cm 15 m from the scanner. A survey
whose positions share few trees may still register only in part. The same
pipeline is the `sylva coreg` command (see [Command line](cli.md)). The
methods it builds on are cited on the [references](../references.md) page.
