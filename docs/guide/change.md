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
| `align_epochs(a, b)` | two epochs | transform on stems and ground, with its uncertainty |
| `match_trees`, `tree_increments` | two stem maps and clouds | survivors, deaths, recruits; increments with a minimum detectable increment |
| `plot_summary(increments)` | one tree comparison | growth, mortality, recruitment with 95 % intervals |
| `compare_qsms(qsm_a, qsm_b)` | two QSMs of a tree | taper increment, branches lost, new and grown; volume change where both were measured |

## Trees and plots

### Worked example

The synthetic plot is 30 m square with 16 trees, scanned from five
positions in each epoch; the second epoch has its own range noise (5 mm
against 3 mm), scanners displaced by about 0.5 m, and is delivered in a
frame rotated by 1.5 degrees and shifted by almost a metre, as an
independently registered revisit would be.

```python
from sylva import change, ground, synthetic, trees

ep = synthetic.forest_epochs(seed=1)
cloud_a, cloud_b = ep.clouds                 # 652,715 and 605,454 echoes


def inventory(cloud):
    """The operational sequence of the trees guide."""
    cloud = ground.normalize_height(cloud, ground.make_dtm(cloud))
    stems = trees.detect_stems(cloud)
    stems, _ = trees.merge_branches(cloud, stems)
    labels = trees.segment_trees(cloud, stems)
    trees.tree_heights(cloud, labels, stems)
    return trees.prune_trees(stems, labels) + (cloud,)


stems_a, labels_a, cloud_a = inventory(cloud_a)
stems_b, labels_b, cloud_b = inventory(cloud_b)

al = change.align_epochs(*ep.clouds)         # epoch 2 onto epoch 1
m = change.match_trees(stems_a, stems_b, transform=al)
inc = change.tree_increments(m, cloud_a, cloud_b, labels_a, labels_b,
                             noise_a=0.003, noise_b=0.005)
summary = change.plot_summary(inc, area=30 * 30, years=5)
print(al.report())
print(summary.report())
```

The alignment matches the 14 surviving stems and 1,448 terrain samples; its
registration sigma is 1.3 mm, and the transform differs from the true one
by 0.1 mm horizontally and 0.9 mm vertically at the plot centre, 1.4 times
its vertical sigma of 0.6 mm. `m.counts()` gives 14 survivors, 2 deaths and
2 recruits, which are exactly the trees that died and grew in; one recruit
stands 0.37 m from the stump of a dead tree and is not mistaken for it. Some rows of `inc.to_pandas()`, with the true
increments beside them:

| True tree | ΔDBH (mm) | true | MDI (mm) | `dbh_change` | Δheight (m) | true |
|---|---|---|---|---|---|---|
| 12 | 6.6 | 6.2 | 1.3 | growth | 0.91 | 0.91 |
| 6 | 15.9 | 16.0 | 1.3 | growth | 0.34 | 0.31 |
| 16 | 18.4 | 18.0 | 2.3 | growth | 0.64 | 0.62 |
| 11 | 5.6 | 5.8 | 1.1 | growth | 0.13 | 0.53 |
| 1 | 0.2 | 0.5 | 2.9 | below_detection | 0.01 | 0.01 |
| 10 | 1.0 | 0.5 | 3.0 | below_detection | 0.01 | 0.01 |

Trees 1 and 10 grew by only 0.5 mm, less than a scan can resolve, and are
reported as below detection rather than as growth. Every height increment
here is below its detection level of about 1 m (tree 11 shows why: one epoch
missed its top by 0.4 m), so the table does not claim height growth even
where the measurement happens to be right. The plot summary gives, per
hectare and year, a basal-area growth of 0.184 m² (95 % interval 0.177 to
0.191; true 0.184), a mortality of 0.405 m² (0.399 to 0.412; true 0.405) and
a net change of −0.187 m² (−0.196 to −0.177; true −0.187). The whole example runs in about
8 s.

### Aligning epochs

Between surveys crowns grow, lose limbs and move in the wind, so a
registration over all points is pulled by the change it should reveal.
`align_epochs` takes a coarse transform from the coregistration pipeline
(`sylva.coreg.register_pair`: a global stem-map match refined by ICP) and
refines it on stable features only:

- **stems**: the horizontal position of every stem found in both epochs
  within `stem_tolerance` (a stem thickens, but its axis stays where it
  was), which fixes x, y and the rotation about z;
- **terrain**: the height of the new terrain model above the reference one,
  which fixes z and the two tilts.

Robust Gauss-Newton with Huber (1964) weights, each feature type weighted by
its own spread, gives the transform and the covariance of its six
parameters. `registration_sigma` is the RMS, over the reference stems, of the
3-D displacement uncertainty this covariance implies: the uncertainty of the
transform, not the residual of one feature (15 stems with 5 mm residuals fix
the translation to about 1.3 mm). Two choices keep it honest. Terrain errors
are spatially coherent, so all terrain samples together weigh as much as one
independent sample per `ground_block` (5 m); and no stem centre is taken as
better than `stem_floor` (2 mm) nor a terrain sample as better than
`ground_floor` (5 mm), whatever an unusually clean plot suggests. Without
these, the synthetic alignment errors exceeded their sigma by factors of two
to eight. `stable="stems"` or `"ground"` uses one kind alone; the other
parameters then stay at the coarse transform and their sigma is NaN.

A project of several scans per epoch is aligned through its registered,
merged cloud (`sylva.coreg.merge_clouds`), or through scans already prepared
with `sylva.coreg.prepare_scan`. `al.apply(cloud)` moves any cloud of the new
epoch into the reference frame.

### Matching trees

`match_trees` moves the second epoch's stems into the first epoch's frame
and pairs them by an optimal assignment (Kuhn 1955; Munkres 1957) that first
matches as many trees as it can and then minimises the summed cost of
distance and relative DBH and height differences. Unlike nearest-neighbour
matching, a stem cannot take its neighbour's partner when that would leave
the neighbour unmatched. A pair is allowed only within `max_distance` (1 m)
and when the DBH grew by at most `dbh_tolerance` (35 %) or shrank by at most
`max_shrink` (15 %): stems do not shrink, so a felled tree and the sapling
beside it are a death and a recruit, not one tree that lost half its
diameter.

Stems left over are checked for merges and splits before they are called
deaths or recruits. When a stem of the second epoch is wide enough to stand
for an unmatched neighbour as well as its own partner (its DBH at least
`merge_factor` times the quadratic sum of theirs), the neighbour is `merged`
into it: the detector saw two touching stems as one. Splits are the mirror
image. `TreeMatch.ambiguous_pairs()` marks survivor pairs involved in either,
whose increments compare a stem with more or less than itself; the plot
summary leaves them out and counts them.

### Increments and the minimum detectable increment

A DBH increment is measured on the stem profile rather than as the
difference of two DBHs. Both epochs' stems are cut into slices from 1 to 3 m
every 0.25 m, a circle is fitted to each with the same inlier distance in
both epochs, and slices far off the taper line of the others (a branch
junction) are dropped. The increment is the weighted mean of the paired
differences at the same heights, which cancels the stem's own shape: a
flattened or fluted stem has the same shape in both epochs.

Its standard error has four parts: the precision of each slice's circle,
`2 s / sqrt(n) / sqrt(c)` (fit residual `s`, at least the epoch's range noise;
`n` inlier points; share `c` of the circumference seen); any excess scatter
of the differences between slices (Birge ratio); a share
`slice_correlation` (0.3) of the slice errors that does not average out,
because the same scanners see every slice of a stem from the same
directions; and the vertical registration uncertainty times the stem taper.
The minimum detectable increment (MDI) is that error times 1.96 (at the
default 95 % confidence). An increment within ±MDI is `below_detection`; a
decrease beyond it is `decrease` and flagged `implausible`, as is an increase
above `max_dbh_increment` when one is given. Range noise can be given per
epoch as a number or as a `sylva.quality.stem_noise` result.

Height is the highest point of the tree above its own epoch's terrain (so
vertical misregistration cancels), continued upwards through unassigned
points within 0.3 m of the stem, because segmentation often loses a thin
leader. Its uncertainty adds `height_error` (2 % of the height) to the
spread of the highest points: a top that no pulse hit leaves no trace in the
epoch's own data. With multiple scans the measured tops are usually right to
a few centimetres, but not always, so a height increment of less than about
a metre on a 20 m tree is reported as below detection. Crown area and
volume increments are given as measured, without a detection level:
occlusion changes them as much as growth does.

### Plot summary

`plot_summary` sums the trees per hectare and year: survivors' basal-area
and stem-volume growth, the basal area and volume of the dead trees (first
epoch) and of the recruits (second epoch), their net, annual mortality and
recruitment rates (Sheil et al. 1995), and biomass when a wood density is
given. Stem volume is `form_factor` (0.5) times basal area times height.
Intervals come from Monte Carlo draws in which every tree's DBH, height and
increments are perturbed by their standard errors; a survivor without a
measured increment takes a measured survivor's increment. They cover
measurement noise and registration only: the counts are taken as exact and
the sampling error of a plot as a sample of a stand is not included.

### Provenance

A difference between epochs is change only if both epochs were measured the
same way; a different stem-detection setting or software version can move a
DBH by more than a year's growth. Record each epoch when it is processed and
compare the records before comparing the epochs:

```python
settings = {"detect_stems": {}, "segment_trees": {"power": 6.0},
            "increments": inc.settings, "alignment": al.settings}
record = change.provenance(settings).records[0]   # store it with the epoch's products
...
change.provenance(record_2019, record_2024)         # warns: ProvenanceWarning
```

Each record holds the Sylva version and the settings; the comparison
flattens nested settings (dataclasses such as `CoregConfig` included) to
dotted keys and lists every key whose value differs, with numbers equal
within `rtol`.

### Validation

Twenty `forest_epochs` plots (seeds 0 to 19, 280 surviving trees) went
through the workflow above:

| Check | Result |
|---|---|
| Deaths and recruits | recovered exactly in 20 of 20 plots, felled-and-replaced trees included |
| DBH increment within its MDI | 280 of 280 trees (median MDI 1.4 mm, largest 3.2 mm) |
| 0.5 mm increments reported as below detection | 40 of 40 |
| Height increment within its MDI (crowns left intact) | 204 of 216 (94 %) |
| Alignment error at the plot centre | RMS 0.11, 0.09 and 1.18 sigma in x, y, z; largest 1.7 sigma |
| 95 % intervals covering the true plot values | basal-area growth, mortality and net change 20, 20 and 19 of 20; volume growth and net change 20 and 19; mean DBH increment 20 |

The synthetic scanner places every echo of a pulse along the direction of
its last one, so with two echoes per pulse the first echoes at a stem's
silhouette are displaced sideways and circles come out a few millimetres
wide, differently in each epoch, much like the mixed-pixel edges of a real
scanner. With `max_echoes=2`, 260 of 280 increments (93 %) lay within their
MDI, and the plot intervals covered the truth in 16 to 20 of 20 plots. The
MDI is therefore not conservative for data with pronounced edge effects:
filter mixed pixels first, or raise `slice_correlation`.

`tests/test_change_trees.py` repeats these checks on smaller plots, and
covers the failure modes: stems that moved by 0.3 m, two stems merged into
one and split again, a tree felled with a new one grown beside it, epochs
processed with different settings, and increments too small to detect.

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
