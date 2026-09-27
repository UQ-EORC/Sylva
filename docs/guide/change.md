# Change detection

`sylva.change` compares two epochs of one plot. Throughout, the aim is to
separate real change from noise and from not having seen something: every
change carries its uncertainty or level of detection, and whatever the data
cannot support is labelled (below detection, unmeasured, ambiguous) rather
than reported as change.

| Function | Compares | Gives |
|---|---|---|
| `align_epochs(ref, new)` | stems and terrain of two epochs | the transform new → ref and its uncertainty |
| `match_trees(trees_a, trees_b)` | stem maps | survivors, deaths, recruits, merges, splits |
| `tree_increments(match, ...)` | the survivors' stems and tops | DBH, height and crown increments with a minimum detectable increment |
| `plot_summary(increments, area)` | the whole plot | growth, mortality, recruitment and net change with intervals |
| `provenance(settings_a, settings_b)` | how each epoch was processed | the differences, and a warning |

`sylva.synthetic.forest_epochs` makes two scanned epochs of one plot with
known changes (growth, deaths, recruits, lost limbs, thinned foliage, and a
known registration offset), against which everything here is validated.

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

## Distances, surfaces and voxels

Point-level distances between epochs with their level of detection, the
difference of two terrain or canopy models, and the change in voxel
occupancy are described in this section as they are added.

## Quantitative structure models

The comparison of two QSMs of one tree, branch by branch, is described in
this section as it is added.
