# TLS and ALS together

A terrestrial scan measures a plot from below: every stem, its diameter,
the understorey and the lower crowns, in millimetre detail but over a
hectare at most, and with the tops of tall trees often hidden. An airborne
survey measures the same forest from above: the upper canopy and the
terrain over the whole landscape, but hardly a stem. `sylva.fusion` puts
the two in one frame and combines them:

| Step | Function | What it gives |
|---|---|---|
| Registration | `fusion.register` | the TLS plot placed on the ALS survey, with its residuals and uncertainty |
| Tree linking | `fusion.link_trees` | TLS stems linked to ALS trees, the trees under each crown, and one table of TLS diameters and ALS heights |
| Merged clouds | `fusion.merge_clouds` | one point cloud, TLS below and ALS above, with the source of every point |
| Merged profiles | `fusion.fuse_profiles` | one plant area density profile, each height taken from the instrument that sampled it better |
| Upscaling | `fusion.plot_values`, `fit_model`, `upscale` | plot values from the TLS regressed on ALS metrics and predicted over the survey, with leave-one-out checks and prediction intervals |

```python
import numpy as np
from sylva import als, fusion, ground, read, registration, trees, voxels

tls = ground.classify_ground_csf(read("plot.laz"))           # the plot, in its scanner frame
cat = als.catalog("tiles/")                                  # the survey, ground classified
gnss = registration.translation(512_340.0, 6_912_870.0, 0.0) # the plot centre, to a few metres

reg = fusion.register(tls, cat, gnss, search_radius=10.0)    # heading searched over 360 degrees
print(reg.report())
tls_map = reg.apply(tls)                                     # the plot in the survey's frame

links = fusion.link_trees(stems, als_trees, reg, volumes=qsms)
links.to_csv("trees.csv")                                    # TLS DBH, ALS height, flags
```

The computations are in the Rust core (`sylva_rs::fusion`); results do not
depend on the number of threads.

## Registration

A TLS plot arrives in one of two states: in a local frame (a scanner or
project frame, placed roughly by a hand-held GNSS position and perhaps a
compass heading), or already georeferenced, with the metre-level error of
the GNSS antenna on the scanner. `fusion.register` handles both; they
differ only in how wide a search they need (`search_radius`,
`heading_range`).

**Search.** Each cloud gets a canopy height model (CHM, the highest point
of each cell above a DTM made from its own ground points) and a terrain
model (DTM, the lowest ground point of each cell). The cells of the TLS
plot form a template that is turned by a heading and shifted over the ALS
rasters, and each pose is scored by Pearson's correlation of the two CHMs,
plus `dtm_weight` (0.5) times that of the two DTMs, over the cells both
measured. A correlation is blind to a vertical offset between the terrains
and to a TLS canopy that reads low where the scanner saw less of the
crowns. The whole window is scored at `coarse_resolution` (2 m): every
heading within `heading_range` of the initial one, `heading_step` (3°)
apart, and every shift within `search_radius`. The `n_candidates` (3) best
separated peaks are searched again at `resolution` (0.5 m), with bilinear
sampling and a parabola through the best score in heading, x and y. For a
sparse survey the fine cells are widened to hold about `returns_per_cell`
(10) returns each: 1 m at 10 returns per m². The height comes from the
median difference of the two DTMs.

**Refinement.** From each peak, a robust point-to-plane ICP (the ICP of
[`sylva.coreg`](preprocessing.md#coregistering-a-survey): Huber weights,
a trimmed tail, a voxel pyramid of 1, 0.5 and 0.25 m) moves the TLS points
onto the ALS points: ground, stems and crowns (`refine="all"`), the ground
alone (`"ground"`), or nothing (`"none"`). A run counts only if it stays
within `max_refine_shift` (3 m) and `max_refine_turn` (8°) of its peak;
of those, the one that fits best is kept (the most TLS points within the
last cut-off of an ALS point, the higher peak when two fit within half a
percent). The 3-D fit of ground and crowns separates poses the canopy
models cannot: a TLS canopy model reads the crowns from below and to the
side, so with one or two scan positions its best correlation can lie a few
degrees or metres from the truth (the validation below has cases of 2 m
and 7°), and the ICP from that peak recovers it.

**Checks and uncertainty.** `reg.report()` lists what can be checked
without a reference:

```text
TLS to ALS registration
  pose       heading +35.00 deg, shift x +1.146 y -3.525 z +1.904 m about (48.86, 53.76, 0.60)
  search     score 0.988 (CHM r 0.983, DTM r 1.000), 11444 cells of 0.5 m, ambiguity 0.07
  icp        accepted: fitness 0.939, rmse 0.268 m, 187886 pairs, moved 0.034 m and 0.03 deg from peak 1
  ground     TLS minus ALS DTM: median +0.040 m, NMAD 0.008 m (204735 points)
  canopy     to the nearest ALS return: median 0.201 m, 90 % 0.353 m (195001 points)
  std error  x 0.008 m, y 0.010 m, z 0.001 m, heading 0.017 deg
```

- *ambiguity* is the misfit (1 minus the score) of the best peak over that
  of the second: 0 for a clear answer, near 1 when another pose fits about
  as well (a warning is printed above 0.8). It is high in uniform stands,
  where the canopy repeats itself.
- *ground* compares the TLS ground points with the ALS DTM: the median is
  the vertical offset left (here the 4 cm by which a DTM of the lowest
  return per cell sits below the surface), the NMAD its spread.
- *canopy* is the distance of TLS canopy points to the nearest ALS return.
- *std error* is the larger, for each of x, y, z and the heading, of two
  estimates. The formal one, `sigma² H⁻¹` from the ICP's point-to-plane
  information (`reg.covariance`, 6 x 6 about the pivot), treats every
  residual as independent and is a lower bound. The jackknife
  (`reg.jackknife`) runs the ICP four more times, each with one quadrant
  of the plot left out, and shows how much the answer depends on which
  part of the plot is used.

Everything is computed about a pivot near the plot (the centre of its
ground), so map coordinates of any size lose no precision; the shifts and
uncertainties refer to that pivot. The TLS is thinned to a tenth of a
fine cell first, so plots of tens of millions of points take seconds: a
plot of 20.6 million points was registered onto 9.6 million ALS returns
in 8 s on eight cores.

Two settings matter most in practice:

- **The search window.** For a plot in a scanner frame, give its centre to
  within `search_radius` in `initial` and leave `heading_range` at 180 (a
  full turn). For a georeferenced plot, `search_radius` a few metres and
  `heading_range` a few degrees (with `heading_step` of 1) are faster and
  leave less room for a wrong peak.
- **The plot's extent.** Crop the TLS to the plot (the stand the scanners
  saw well) rather than every distant return; at least half of its canopy
  cells must fall on ALS data (`min_overlap`).

## Tree linking

`fusion.link_trees` links the TLS trees (stem positions, DBH, heights,
volumes) to the ALS trees (tops, heights, crown outlines, from
[`als.segment_trees` or `als.find_trees`](als_trees.md)). A stem and an
ALS tree can be one tree when the stem lies inside the crown (or within
`crown_buffer`, 0.5 m, of its outline) or within `max_distance` (3 m) of
the top. Among these pairs an optimal assignment (Kuhn 1955; Munkres 1957)
minimises the total cost, a stem left without an ALS tree costing
`max_cost` (2). A pair costs

```text
(d / D)² + height_weight * dh² + dbh_weight * (1 - dbh / dbh_max)²
```

with `d` the stem-to-top distance, `D` the larger of `max_distance` and
the crown's equivalent radius, `dh = (h_als - h_tls) / h_als` (0.3 for a
stem without a TLS height), and `dbh_max` the largest DBH among the stems
that could be that tree (both weights 2). The stem that is tallest by the
TLS and thickest under a crown is therefore its tree, as the dominant tree
of a crown usually is both; the diameter term matters because TLS heights
are unreliable exactly where it matters, in the upper crowns. Each ALS
tree gets at most one stem.

The stems under a crown that are not its tree are reported explicitly:

| Status | Meaning |
|---|---|
| `matched` | the stem is this ALS tree |
| `suppressed` | under the crown of a taller ALS tree (by more than `top_tolerance`, 1 m, and 10 %): an understorey or overtopped tree the ALS cannot see |
| `codominant` | under an ALS crown whose tree is another stem of about the same height: the ALS saw two canopy trees as one |
| `unlinked` | under no ALS crown and near no top |

`links.one_to_many()` gives each ALS crown that holds two stems or more,
with the TLS ids under it (its own stem first), and `links.als_table()`
every ALS tree with its stems.

**The combined table.** `links.table()` has one row per TLS tree: its DBH,
TLS height and volume, its status and ALS tree, and a combined `height`
with its `height_source` and a `flag`:

- a matched tree takes the ALS height, which sees the top from above;
  unless the TLS is known to have seen the top too and measured it
  taller (both are lower bounds, but a TLS height can also be too tall
  where the segmentation gave a tree part of a neighbour's crown);
- other trees keep the TLS height, flagged `top_not_seen` when the TLS
  did not see the top, so that the height is a lower bound.

Whether the TLS saw a top comes from `top_seen`, or from `sampling`, the
output of [`voxels.tree_sampling`](voxels.md#what-the-scan-saw): a top
counts as seen when `above_observed_fraction`, the share of the space just
above the tree's highest point that the scan observed, is at least
`min_above_observed` (0.5). Without either, a matched tree's top counts as
seen when its TLS height reaches the ALS height within `top_tolerance`.

```python
g = voxels.ray_voxelize(shots, 0.25, dtm=dtm, occlusion=True)
sampling = voxels.tree_sampling(g, tls, labels)
links = fusion.link_trees(stems, als_trees, reg, volumes=qsms, sampling=sampling)
links.one_to_many()            # {als id: [tls ids]}
```

## One cloud, one profile

`fusion.merge_clouds(tls, als, dtm, split)` keeps the TLS points below a
split height above ground and the ALS returns above it, and marks each
point's `source` (1 TLS, 2 ALS) and `height`. The split is a height, a
raster of heights (one per column), or a fused profile, whose
`split_height` is the lowest height above which the ALS carries at least
half the weight.

`fusion.fuse_profiles(tls_grid, als_grid, dtm, area)` combines the
ray-traced voxels of both instruments (a TLS
[`RayVoxelGrid`](voxels.md) and an ALS
[`ALSVoxels`](als_canopy.md#ray-traced-voxels)) into one plant area density
(PAD) profile over the same area (a box, a circle or a polygon). Each grid
is summarised per bin of height above the DTM: the PAD of the bin, the mean
number of pulses entering its voxels (voxels no pulse reached count 0, so
occlusion lowers it) and the share of its voxels crossed by at least
`min_beams` (5) pulses. Each bin then takes its PAD from both instruments,
weighted by

- `"beams"` (the default): the pulses, since the variance of a
  gap-fraction estimate falls in proportion to the pulses that sampled it;
- `"observed"`: the share of the bin each instrument saw;
- `"best"`: the one with more pulses alone.

The weights are part of the result (`weight_tls`, `weight_als`, and each
instrument's own profile in `tls` and `als`). The TLS pulses must first be
moved into the survey's frame, with the registration, and both grids should
share the voxel size:

```python
g_tls = voxels.ray_voxelize(shots.transform(reg.transform), 1.0, box, ground_class=2)
g_als = als.ray_voxelize(cat, traj, voxel_size=1.0, z_range=(zmin, zmax))
prof = fusion.fuse_profiles(g_tls, g_als, dtm, area=(x0, y0, 15.0))
print(prof.pai("tls"), prof.pai("als"), prof.pai())
prof.table()                   # height, pad, weights, and each instrument's profile
```

**The estimator.** By default a bin's PAD is the mean of the voxels' own
estimates (`estimator="mean"`, the field `pad_fpl`), over the voxels
observed, so a crown and the gap beside it count by their volume. The
alternative, `"pooled"`, divides the bin's weighted hits by its free path
length; that ratio of sums is not biased upwards by poorly sampled voxels,
but it weights each voxel by the pulses that reached it, and in a clumped
canopy the shadowed crowns receive fewer pulses than the gaps between them.
On an open stand of 90 trees like those of the validation below, the
pooled estimate read the plant area 36 % low from the TLS and 25 % low
from the ALS, the mean 4 % low and within 1 %; use `"pooled"` only for
horizontally uniform layers.

## Upscaling

The area-based approach (Næsset 2002) relates plot values measured on the
ground to ALS metrics of the same plots, and predicts across the survey
from a raster of the same metrics:

```python
values = [fusion.plot_values(plot_trees, area=706.9, wood_density=600.0) for plot_trees in plots]
agb = np.array([v["agb"] for v in values])                       # Mg/ha
pm = als.plot_metrics(cat, centres, radius=15.0, metrics=["zq95", "cover"], min_height=0.0)
grid = als.grid_metrics(cat, 25.0, ["zq95", "cover"], min_height=0.0)
up = fusion.upscale(agb, pm, grid, ["zq95", "cover"], model="loglog")
print(up.model.summary())
up.mean.to_geotiff("agb.tif"); up.se.to_geotiff("agb_se.tif")
```

- `fusion.plot_values` sums a plot's trees per hectare: above-ground
  biomass (wood volume, e.g. from QSMs, times the wood density; there is no
  default density), volume, basal area and stem density, above an
  inventory `min_dbh`.
- `fusion.fit_model` fits ordinary least squares, `"loglog"` (`ln y = b0
  + Σ bk ln xk`, a power law, the usual form for biomass) or `"linear"`.
  Log-log predictions are taken back as `exp(ŷ + σ²/2)` (Baskerville
  1972), which estimates the mean rather than the median.
- Leave-one-out predictions come from the hat matrix in closed form
  (equal to refitting without each plot, which the tests check), and give
  the RMSE, bias and R² a new plot can expect (`model.loo_rmse`,
  `loo_bias`, `loo_r2`, `loo_rrmse`). Prefer them to the in-sample R²
  when choosing predictors.
- `fusion.upscale` predicts every cell: the mean, the standard error of
  a new observation, `σ √(1 + x₀ᵀ (XᵀX)⁻¹ x₀)`, and the Student t interval
  at `level` (for log-log, the interval of `ln y` taken back and the
  standard error of the log-normal distribution it implies). Cells whose
  metrics lie outside the range of the plots are marked in
  `up.extrapolated`.

Keep the model small: one or two metrics for ten or twenty plots, since
each costs a degree of freedom and correlated metrics (`zq95` and `zmax`)
add little; and compute the grid at cells of the plots' area, since height
percentiles depend on the area they summarise.

## Synthetic scenes seen by both

`fusion.synthetic_scan` is a terrestrial scanner for the synthetic scenes
that sees them as [`synthetic.als_flight`](als.md#the-synthetic-scanner)
does: every scene point is a sphere of `target_radius` and the ground is
the analytic terrain, and each pulse is a thin ray that stops at the first
sphere or the terrain. A layer of spheres of `n` per m³ is then a turbid
medium of plant area density `2 π r² n` for both scanners, so the TLS and
ALS estimates can be checked against the same known foliage.
([`synthetic.scan`](../api/synthetic.md) instead lets any point in a
pulse's angular cell be hit, which suits geometry but not densities.)

## A worked example

One forest, flown over at about 95 returns per m² and scanned from five
positions; the TLS is given in a frame turned by 35° whose centre is known
to within 4 m.

```python
import numpy as np
from sylva import PointCloud, Shots, als, filters, fusion, ground, registration, synthetic, trees

stand = synthetic.stand(90, size=100.0, min_spacing=3.0, heights=(8, 25), seed=3)
scene = synthetic.crown_forest(stand, size=100.0, ground_points=100, margin=0.0, seed=3)
flight = synthetic.als_flight(scene, pulse_rate=20_000, line_spacing=30.0, bounds=(0, 0, 100, 100), seed=3)
origins = [(x, y, float(synthetic.terrain_height(x, y)) + 1.5)
           for x, y in [(50, 50), (38, 38), (62, 38), (38, 62), (62, 62)]]
shots = fusion.synthetic_scan(scene, origins, resolution_deg=0.2)

truth = registration.rotation_z(35.0)
truth[:3, 3] = [53.1, 47.8, 1.9]
local = Shots.concatenate(shots).transform(np.linalg.inv(truth))   # what the scanners recorded
tls = local.to_pointcloud()
c = np.linalg.inv(truth) @ [50, 50, 0, 1]
tls = tls[np.hypot(tls.x - c[0], tls.y - c[1]) < 30]                # the plot

reg = fusion.register(tls, flight.points, registration.translation(50, 50, 0), search_radius=8.0)
err = reg.transform @ np.linalg.inv(truth)
print((err @ [50, 50, 0, 1])[:3] - [50, 50, 0])                    # [-0.002  0.008  0.004]

tl = filters.voxel_downsample(tls, 0.02)
tl = ground.normalize_height(tl, ground.make_dtm(tl, 0.5))
stems = trees.detect_stems(tl)
stems, _ = trees.merge_branches(tl, stems)
labels = trees.segment_trees(tl, stems)
trees.tree_heights(tl, labels, stems)
stems, labels = trees.prune_trees(stems, labels, min_height=3.0)
pts = flight.points
dtm = ground.make_dtm(pts, 1.0)
tops = als.segment_trees(pts, heights=ground.normalize_height(pts, dtm).attrs["height"],
                         window=als.LinearWindow(0.0, 0.2, 2.0, 20.0), max_cr=20)
links = fusion.link_trees(stems, tops, reg)
print(links)      # TreeLinks(32 TLS trees: 20 matched, 10 suppressed, 2 codominant, 0 unlinked)
merged = fusion.merge_clouds(reg.apply(tls), pts, dtm, split=10.0)
```

The report of this registration is the one shown above; the whole example
runs in about 15 s.

## Validation

All of it on `synthetic.crown_forest` stands (ellipsoidal crowns of radius
a quarter of the tree height, filled with leaf points; heights 8 to 25 m)
flown by `synthetic.als_flight` and scanned by `fusion.synthetic_scan`, so
that the trees, the pose and the plant area are known.

**Registration.** One hectare with 90 trees, scanned from five positions
at 0.2° (with 1 cm noise added to the TLS points, cropped to 30 m around
the plot centre) and flown at three pulse rates. The local frame was
turned by a random heading (full circle searched) and the plot centre given
to within 4 m; the georeferenced plot was off by up to 3 m and 3° (searched
within 5 m and 5°). Three stands per density; the errors are at the plot
centre after the search alone and after the ICP, and the standard errors
are those `reg.std` reports.

| ALS returns/m² | Case | Search error: horizontal (m), heading (°) | Final error: horizontal (m), vertical (m), heading (°) | Reported std error x, y (m) |
|---|---|---|---|---|
| 95 | local | 0.01 to 0.06, 0.05 to 0.15 | 0.005 to 0.012, 0.003 to 0.004, 0.001 to 0.015 | 0.003 to 0.009 |
| 95 | georeferenced | 0.04 to 0.06, 0.02 to 0.18 | 0.005 to 0.009, 0.002 to 0.006, 0.004 to 0.015 | 0.004 to 0.009 |
| 10 | local | 0.02 to 0.13, 0.29 to 0.44 | 0.004 to 0.042, 0.000 to 0.003, 0.03 to 0.14 | 0.011 to 0.034 |
| 10 | georeferenced | 0.10 to 0.16, 0.04 to 0.62 | 0.019 to 0.031, 0.000 to 0.003, 0.003 to 0.14 | 0.009 to 0.041 |
| 2 | local | 0.25 to 0.92, 0.28 to 1.19 | 0.052 to 0.067, 0.001 to 0.002, 0.01 to 0.05 | 0.034 to 0.073 |
| 2 | georeferenced | 0.15 to 1.13, 0.10 to 2.72 | 0.069 to 0.145, 0.002 to 0.005, 0.03 to 0.25 | 0.030 to 0.110 |

The ICP brings every case to within 15 cm, and to a centimetre at dense
point spacing; the reported standard errors are of the size of the actual
errors, within a factor of about three either way. On smaller stands (0.36
ha, 30 trees, two scan positions at 0.4°, three stands, the TLS cropped to
18 m and to 25 m), the search alone was 1.5 to 2.7 m and 2 to 7.6° off in
all six runs, the TLS canopy model being read from so few positions; the
ICP from the peaks still ended within 1.5 cm in all six. The ground-only
ICP (`refine="ground"`) fixes the height and tilt but can drift sideways on
gentle terrain (6 cm and 29 cm in two runs on the first stand above), so it
is best kept for bare plots.

**Tree linking.** Three such stands, registered as above, with the TLS
trees from `trees.detect_stems`, `merge_branches`, `segment_trees`,
`tree_heights` and `prune_trees`, and the ALS trees from
`als.segment_trees` (Dalponte 2016, a window of a fifth of the height,
`max_cr=20`). Of the 44 TLS trees within 25 m of the plot centres, 30 had
their own ALS tree (the others are overtopped and form no local maximum in
the canopy surface):

| | Count | Right |
|---|---|---|
| TLS trees linked to their own ALS tree | 30 of 30 | 30 |
| Links to another tree's ALS crown | 0 | |
| Reported `suppressed` or `codominant` | 13 | 13 (none had its own ALS tree) |
| `unlinked` | 1 | |

The heights of the linked trees against the truth: the TLS alone had a
mean error of -1.19 m and an RMSE of 3.46 m (its segmentation gives crowns
wrongly in these closed canopies, and misses tops); the combined height a
mean error of +0.03 m and an RMSE of 0.07 m. The TLS DBH was within 1.5 cm
(RMSE).

**Profiles.** The same kind of stands, of leaves alone (the stems of the
scene are points on a surface, not a turbid medium, and are left out of the
truth), their true PAD from the number of spheres per height bin; voxels of
1 m, PAD by height within 15 m of the plot centre. PAI and the RMSE of the
PAD bins against the truth:

| Stand | True PAI | TLS (five scans) | ALS (7 sub-beams) | Fused, beams | Fused, observed | Fused, best |
|---|---|---|---|---|---|---|
| open, 90 trees, seed 1 | 0.42 | 0.42 (0 %, 0.001) | 0.47 (+12 %, 0.004) | 0.42 (0 %, 0.001) | 0.44 (+6 %, 0.002) | 0.42 (0 %, 0.001) |
| open, 90 trees, seed 2 | 0.77 | 0.76 (-1 %, 0.001) | 0.79 (+3 %, 0.003) | 0.76 (-1 %, 0.001) | 0.77 (+1 %, 0.002) | 0.76 (-1 %, 0.001) |
| dense, 200 trees, seed 1 | 5.80 | 5.57 (-4 %, 0.017) | 4.70 (-19 %, 0.091) | 5.52 (-5 %, 0.018) | 5.16 (-11 %, 0.049) | 5.57 (-4 %, 0.017) |
| dense, 200 trees, seed 2 | 5.67 | 5.40 (-5 %, 0.022) | 4.42 (-22 %, 0.092) | 5.37 (-5 %, 0.023) | 4.93 (-13 %, 0.053) | 5.40 (-5 %, 0.022) |
| dense, with stems | 6.01 | 5.66 (-6 %, 0.020) | 4.73 (-21 %, 0.083) | 5.61 (-7 %, 0.021) | 5.22 (-13 %, 0.046) | 5.66 (-6 %, 0.020) |
| dense, one TLS scan, 20 m | 6.01 | 5.06 (-16 %, 0.061) | 4.97 (-17 %, 0.073) | 5.16 (-14 %, 0.052) | 5.06 (-16 %, 0.057) | 5.10 (-15 %, 0.060) |
| the same, ALS one ray per pulse | 6.01 | 5.06 (-16 %, 0.061) | 4.90 (-18 %, 0.105) | 5.21 (-13 %, 0.049) | 5.09 (-15 %, 0.061) | 5.12 (-15 %, 0.061) |

With five scan positions the TLS samples every bin with hundreds to
thousands of pulses per voxel, the pulse weighting follows it, and the
fused profile is as good as the better instrument. From a single scan
position the TLS reads the upper crowns low (by 30 % at 18 m and by more
than half above 21 m: it sees them through the lower ones) while the ALS
reads them 13 to 16 % low; there the ALS takes 37 to 51 % of the weight
above 18 m, and the fused profile has the lowest RMSE of the three. The
ALS reads the lower half of dense crowns 30 to 55 % low, because the
voxels deep in a crown are reached by few pulses from above. None of the
weightings can remove a bias both instruments share.

**Upscaling.** A 4 ha landscape of 16 stands of 50 m with 8 to 50 trees
and height ranges between 6 and 26 m, flown at about 20 returns per m²;
15 plots of 625 m² at random, their biomass from the true trees (a stem
volume of half the cylinder of DBH and height, times 600 kg/m³), fitted on
the metrics of `als.plot_metrics` and predicted on a 25 m grid of
`als.grid_metrics`; then compared with the true biomass of every cell.

| Stand | Model | LOO RMSE (plots) | Cells: RMSE, bias (Mg/ha) | Cells: R² | 95 % interval covers |
|---|---|---|---|---|---|
| 1 | loglog, zq95 | 42 % | 29.5 (69 %), +9.9 | 0.24 | 82 % |
| 1 | loglog, zq95 + cover | 12 % | 9.4 (22 %), +3.1 | 0.92 | 77 % |
| 1 | linear, zmean | 25 % | 17.1 (40 %), +3.5 | 0.74 | 97 % |
| 2 | loglog, zq95 | 72 % | 23.4 (54 %), +6.1 | 0.25 | 100 % |
| 2 | loglog, zq95 + cover | 32 % | 8.2 (19 %), +0.1 | 0.91 | 95 % |
| 2 | linear, zmean | 43 % | 12.8 (29 %), 0.0 | 0.78 | 100 % |
| 3 | loglog, zq95 | 46 % | 15.3 (40 %), -4.0 | 0.61 | 91 % |
| 3 | loglog, zq95 + cover | 30 % | 7.0 (17 %), -0.8 | 0.91 | 98 % |
| 3 | linear, zmean | 25 % | 7.7 (20 %), -2.4 | 0.90 | 99 % |

Height alone cannot tell a sparse stand of tall trees from a dense one;
height and cover together recover the landscape's biomass cell by cell
(R² 0.91 to 0.92), and the leave-one-out RMSE of the plots ranks the
models as the cells do. With 15 plots the intervals cover 77 to 100 % of
the cells. On an exact power law the fit recovers the coefficients to
1e-9, and with noise its leave-one-out predictions, coefficients and
standard errors equal refitting and NumPy's least squares (tests).

## Command line

```bash
sylva fusion-register plot.laz tiles/ reg.json --initial gnss.txt --search-radius 10 \
    --transformed plot_map.laz
sylva fusion-trees tls_trees.csv als_trees.csv linked.csv --crowns crowns.geojson \
    --registration reg.json --als-output als_linked.csv
sylva fusion-upscale plots.csv metrics/ agb/ --response agb --predictors zq95,cover
```

`fusion-register` reads the TLS cloud (ground classified) and a directory
of ALS tiles or one file, writes the registration as JSON (and with
`--transformed` the moved cloud) and prints the report; `--initial` is a
text file of the 4 x 4 matrix. `fusion-trees` links a CSV of TLS trees
(`x`, `y` and optionally `tree_id`, `dbh`, `height`, `volume`) to the CSV
and GeoJSON written by `als-trees`. `fusion-upscale` reads a CSV of plots
(the value and the metrics, e.g. the output of `als-plot-metrics` with a
column added) and the rasters of `als-metrics`, and writes the prediction
rasters. See [Command line](cli.md).

## Limitations

- The search needs canopy or terrain structure: in a uniform plantation
  or on flat ground under a closed, even canopy, poses a row apart score
  alike (`ambiguity` near 1); give a tighter `search_radius` there.
- Only a rigid transform is estimated; a TLS project that is itself
  deformed (a poorly registered multi-scan plot) will not fit everywhere.
- The formal covariance is a lower bound, and the jackknife over four
  quadrants is a coarse estimate; neither includes an error the ALS itself
  carries (its own georeferencing).
- Tree linking can only be as good as the two segmentations: ALS crowns
  that merge trees make `codominant` stems, and a TLS tree whose height is
  badly wrong can be left `suppressed` under its own crown.
- The profiles weight by sampling, not by bias: where both instruments
  underestimate a layer (the lower half of dense crowns from the air, the
  upper crowns from a single scan position) the fused profile does too.
- The upscaling models are deliberately simple (ordinary least squares,
  no variable selection, no spatial correlation); the prediction interval
  is that of a single cell, not of a total over many cells.
