# Change detection

`sylva.change` compares two epochs of one plot. Throughout, real change is
kept apart from noise and from not having seen something: every change
carries an uncertainty or a level of detection, and what the data cannot
support is labelled (for example *unobserved*) rather than reported as
change.

| Function | Compares | Result |
|---|---|---|
| `distances(a, b, "m3c2")` | two point clouds | signed distance per core point with its 95 % level of detection |
| `distances(a, b, "c2c")` | two point clouds | distance from each point of `b` to the nearest point of `a` |
| `dod(raster_a, raster_b)` | two DTMs or CHMs | raster of difference, significance mask, volumes |
| `occupancy(grid_a, grid_b)` | two ray-traced voxel grids | gained, lost, stable and unobserved voxels; PAD change |

## Aligning epochs

## Matching trees and growth increments

## Plot summaries

## Surface change: M3C2 and C2C

`distances(a, b, "m3c2", core_points, normal_scale, projection_scale,
max_depth, registration_sigma)` implements the Multiscale Model to Model
Cloud Comparison of Lague, Brodu and Leroux (2013). At each core point a
plane is fitted to the points of the reference epoch `a` within
`normal_scale / 2`; the points of both epochs inside a cylinder of diameter
`projection_scale` along that normal (reaching `max_depth` either way) are
projected onto it, and the distance is the difference of their mean
positions. Its 95 % level of detection is

    LoD95 = 1.96 * (sqrt(s_a² / n_a + s_b² / n_b) + reg)

where `s` is the spread of each epoch's positions along the normal (surface
roughness plus range noise), `n` their number and `reg` the registration
error, and a distance is significant when it exceeds it. On an unchanged
surface about 5 % of the core points are therefore significant by chance
when `registration_sigma` is 0, and fewer when it is not. The scales are
diameters, as in the paper; `normal_scale` should span the surface's
roughness (Lague et al. suggest 20 to 25 times it) and `projection_scale`
should hold at least about 30 points of each epoch.

Normals are oriented upwards by default, which suits ground and other
near-horizontal surfaces. For stems, orient them towards the scanner
(`orientation=("towards", scanner_xyz)`) or pass radial `normals`, so that
radial growth comes out positive.

```python
import numpy as np
import sylva
from sylva import PointCloud, change, filters, synthetic

rng = np.random.default_rng(0)
def terrain(n, deposit=0.0):
    xy = rng.uniform(0, 20, (n, 2))
    z = synthetic.terrain_height(xy[:, 0], xy[:, 1]) + rng.normal(0, 0.005, n)
    z[np.hypot(xy[:, 0] - 10, xy[:, 1] - 10) < 3] += deposit  # a disc of deposition
    return np.column_stack([xy, z])

a, b = terrain(500_000), terrain(500_000, deposit=0.03)
core = filters.voxel_downsample(PointCloud(a), 0.5)          # 1,782 core points
d = change.distances(a, b, "m3c2", core, normal_scale=0.5, projection_scale=0.3,
                     max_depth=0.2, registration_sigma=0.002)
np.nanmedian(d.lod)                    # 0.0054 m
d.distance[d.significant].mean()       # 0.030 m: the 3 cm deposit, and nothing else
sylva.write(d.to_cloud(), "m3c2.laz")  # distance, lod, significant, normals, counts
```

Inside the disc every core point is significant with a mean distance of
30.0 mm; outside it none is (the registration error of 2 mm raises the level
of detection above the chance fluctuations).

`distances(a, b, "c2c")` returns, for each point of `b`, the distance to the
nearest point of `a`. It needs no parameters, but it is unsigned and biased
upwards by point spacing and noise, so it serves as a first look rather than
as a test of change.

Every core point is computed on its own, in parallel, with k-d trees on both
epochs. Two clouds of 20 million points with a million core points take
about half a minute on eight cores, and the results do not depend on the
number of threads.

Agreement with an independent implementation: on a rough synthetic surface
(900 core points, about 120 points per cylinder), the distances match those
of py4dgeo 1.2.0 to 1e-15 m, the levels of detection to 3e-8 m, and the point
counts exactly.

## Rasters of difference

`dod(raster_a, raster_b, min_detectable=None, sigma_a=None, sigma_b=None)`
subtracts two surfaces on the same lattice (a DTM or CHM of difference) and
marks the cells whose change exceeds the level of detection: either
`min_detectable`, or `1.96 * sqrt(sigma_a² + sigma_b²)` from the standard
deviations of the two surfaces (numbers or rasters). Include the vertical
registration error in the sigmas. The result also gives the volumes raised
and lowered over the significant cells.

```python
from sylva import interpolate

dtm_a = interpolate.grid(PointCloud(a), 0.5, method="tin", bounds=(0, 0, 20, 20))
dtm_b = interpolate.grid(PointCloud(b), 0.5, method="tin", bounds=(0, 0, 20, 20))
dd = change.dod(dtm_a, dtm_b, sigma_a=0.005)   # sigma_b defaults to sigma_a
dd.net_volume, dd.area_changed                 # 0.83 m³ (true 0.85), 30.0 m² (true 28.3)
dd.thresholded().to_geotiff("dod.tif")         # change where significant, 0 elsewhere
```

## Voxel occupancy change

A voxel that holds no echoes in the later epoch has only lost its contents
if pulses went through it. `occupancy(grid_a, grid_b)` compares two grids
from `sylva.voxels.ray_voxelize` (pulses of both epochs in one frame, traced
with the same `voxel_size` and `bounds`) and classes every voxel:

| Class | Epoch a | Epoch b |
|---|---|---|
| `stable_empty` | empty | empty |
| `stable_occupied` | occupied | occupied |
| `gained` | empty | occupied |
| `lost` | occupied | empty |
| `unobserved` | not observed in one or both epochs, or too few pulses to tell | |

A voxel is occupied with at least `min_hits` echoes, and empty with none and
at least `min_pulses` pulses entering it; otherwise it was not observed (no
pulse reached it, or only pulses already stopped did). A change between
occupied and empty must also pass a test: if the occupied epoch saw a share
`p` of the entering pulses stopped, the `n` pulses of the empty epoch would
all have missed contents as dense with probability `(1 - p) ** n`, which must
not exceed `alpha` (0.05). Sparse contents grazed by a few pulses are thus
unobserved rather than lost. Plant area density change is given per voxel
and, in `layers`, as the mean per layer over the voxels both epochs sampled
with at least `min_pulses` pulses, so that what one epoch did not see does
not bias the comparison.

```python
scene = synthetic.forest(seed=1)
box = np.all((scene.xyz >= [4, 3, 9]) & (scene.xyz <= [7, 6, 12]), axis=1)
later = scene[~(box & (scene.attrs["classification"] == 4))]   # a block of foliage falls

bounds = ((-2, -2, -1), (22, 22, 19))
grids = [sylva.voxels.ray_voxelize(synthetic.scan(s, origin, 0.25, max_echoes=1), 0.5, bounds,
                                   ground_class=2, occlusion=True)
         for s, origin in ((scene, (10, 10, 1.5)), (later, (10.15, 9.9, 1.5)))]
occ = change.occupancy(*grids, min_pulses=10)
# Occupancy(48x48x40 @ 0.5 m: unobserved=740, stable_empty=86,756,
#           stable_occupied=4,575, gained=6, lost=83)
occ.centers("lost")                    # almost all inside the 3 x 3 x 3 m block
occ.layers["pad_change"]               # -0.002 to -0.003 m² m⁻³ in the layers from 9 to 12 m
```

In a test where the later epoch also has an opaque screen between the
scanner and part of a crown, none of the hidden crown voxels is called lost:
220 of 222 are unobserved and the other two, at the edge of the shadow, are
stable. Of the voxels the fallen block emptied, 75 of 98 are lost and the
rest, holding one to three echoes of 150 or more pulses in the earlier
epoch, are unobserved; none is called stable. In unchanged vegetation 0.3 %
of the occupied voxels come out lost (4 % without the miss-probability
test).

## QSM change

`compare_qsms` compares two cylinder models of one tree, the earlier `a`
and the later `b`, in one frame (align the epochs first). It reports the
stem radius change by height, branch matching with lost, new and grown
branches, volume change by branch order, and height, DBH and crown change.

```python
from sylva import change, qsm, voxels

model_a = qsm.build_qsm(wood_2020, base_xy=(t.x, t.y))
model_b = qsm.build_qsm(wood_2025, base_xy=(t.x, t.y))       # already aligned to 2020
grid_b = voxels.ray_voxelize(shots_2025, 0.1, occlusion=True)  # every 2025 pulse, misses included

c = change.compare_qsms(model_a, model_b, grid_b=grid_b)
c.taper_increment, c.taper_sigma          # mean stem radius increment (m) and its uncertainty
c.taper["increment"], c.taper["trusted"]  # the increment profile, one value per 1 m of height
c.trusted_change, c.untrusted_change      # volume change (m3) both models measured, and the rest
c.orders                                  # the same split for the stem and each branch order
c.branches("lost")["volume"].sum()        # branches gone, their space seen empty in 2025
c.branches("unobserved")                  # missing from the 2025 model, but 2025 did not see there
c.summary()
```

**Trust.** A cylinder is measured when its radius was fitted to points
(`n_points > 0`); the rest of a model comes from the taper and pipe-model
priors of [`build_qsm`](qsm.md) and says little about change. A stem bin is
trusted when both models fitted at least `min_fits` cylinders and
`min_measured` of its length, and when its increment is in line with the
other bins (within `clip` of its own uncertainty from the mean); a branch
pair is trusted when both models measured `min_measured` of its length. The
volume change of every trusted part is `trusted_change`, the rest
`untrusted_change`; they add up to the total, per branch order as well.

**Stem.** In each height bin a line through each model's measured stem
radii gives the radius at the bin centre; their difference is the radius
increment. Its uncertainty is the scatter about both lines plus
`radius_sigma` (1 mm per fit by default), not divided by the number of
cylinders, since the QSM smooths neighbouring radii together. The taper
increment is the weighted mean over the trusted bins, its uncertainty
scaled up by the Birge ratio when the bins scatter more than their
uncertainties allow. A fitted bin far out of line is almost always a
radius the QSM regularised, typically the stem tip tapered to its apex;
it is reported, left out of the mean and counted as untrusted volume.

**Branches.** Branches are matched one order at a time, parents first. A
pair is admissible when the bases lie within `max_base_distance` (0.5 m)
and the directions over the first metre within `max_angle` (35 degrees);
cheaper pairs (distance and angle, each over its limit, plus
`parent_penalty` when the parents are not matched to each other) are taken
first. A matched pair reports its length, volume and radius change and the
shift of its tip. An unmatched branch of the earlier model is lost, one of
the later model new.

**Lost or unobserved.** A branch missing from the later model has either
gone or was not seen. With a ray-traced grid of the later epoch
([`ray_voxelize`](voxels.md), `occlusion=True`) each lost branch is
sampled every half voxel along its axis, leaving out the part inside its
parent: when fewer than `min_observed` of the samples lie in voxels the
later pulses crossed or ended in, the branch is `"unobserved"`; when more
than `max_filled` of the observed samples hold returns, the branch is
`"present"` (still there, but not in the later model); otherwise it is
`"lost"`, and only then does its volume count as trusted loss. A grid of
the earlier epoch (`grid_a`) checks new branches the same way. An
unobserved branch also leaves the crown change untrusted, since the crown
is outlined by the branches.

For a plot, `compare_plot_qsms` pairs the models of matched trees and
tabulates the changes. The tree match is plain data: survivor pairs of tree
ids, and lists of dead and recruited trees.

```python
plot = change.compare_plot_qsms(
    qsms_2020, qsms_2025,                  # PlotQSMs or {tree_id: QSM}
    pairs=[(1, 4), (2, 5), (3, 7)],        # survivors: (id in 2020, id in 2025)
    deaths=[6], recruits=[9],
    grid_b=grid_2025,
)
plot.totals                                # growth (and its trusted part), mortality, recruitment, net
plot.to_csv("qsm_change.csv")              # one row per tree
plot.changes[(1, 4)].taper                 # the full comparison of a survivor
```

A dead tree counts its whole earlier volume as mortality and a recruit its
whole later volume as recruitment, the measured part as trusted.

**Validation.** A synthetic tree (a 10 m tapered stem and five limbs as
point surfaces) was scanned from three positions in each epoch with
`synthetic.scan` (single returns, so foliage is opaque). Between the
epochs the stem thickened by 10 mm along its length, one limb was cut, one
grew 1 m longer, and one was wrapped in a sleeve of foliage that hid it
from every later scan. Over five noise draws:

| Quantity | Truth | Recovered |
|---|---|---|
| Taper increment | 10 mm | 10.1 to 10.3 mm, stated uncertainty 0.6 mm |
| DBH change | 20 mm | 20.1 to 20.8 mm |
| Cut limb | lost | lost in every draw; volume +8 % (the QSM's own bias) |
| Extended limb | +1.00 m | matched, +0.95 to +1.00 m |
| Hidden limb | unobserved | unobserved in every draw; its volume untrusted |
| Stem volume change over the trusted bins | 0.065 to 0.069 m³ | within 7 % |
| Trusted volume change | 0.056 to 0.060 m³ | within 0.8 of the stated uncertainty (5 to 8 L) |

The main error in the trusted total is a limb whose weak fits the QSM
replaced by its allometric prior in one epoch but not the other: its
volume changed by 5 to 7 L although the limb did not change, and the
cylinders still count as measured because the QSM keeps their point count
when it replaces the radius. Treat trusted growth of thin branches that
exceeds its `volume_sigma` by far with care.

Limitations: both models must be in one frame, and the matching is greedy
rather than a global assignment, so a branch that forked differently in
the two models can show as one lost and two new branches. Heights, lengths
and crowns come from the skeleton, which always follows points; the
measured label concerns radii, and so volumes and the taper increment.
Height is trusted when a measured cylinder reaches within `top_band` of the
top in both models, the crown when both models measured most of their
branch length and no unmatched branch went unobserved.
