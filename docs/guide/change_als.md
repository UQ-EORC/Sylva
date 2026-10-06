# Change between airborne surveys

`sylva.change` compares two airborne lidar surveys of one area as it
compares two terrestrial epochs ([Change detection](change.md)): every
change carries its level of detection, and what the data cannot support
is labelled (below detection, no data, uncertain, undetected, unobserved)
rather than reported. The surveys are catalogues of tiles
([Airborne lidar tiles](als.md)); every function runs through the chunk
engine, reading the same buffered box from both surveys, so any area can be
processed and the results do not depend on how it is divided.

| Function | Compares | Result |
|---|---|---|
| `align_surveys(a, b)` | stable surfaces (ground, roofs, roads) | offsets `(dx, dy, dz)` per block, as one offset or a smooth field, with their uncertainty |
| `chm_change`, `dtm_change`, `surface_change` | the highest return per cell, or the terrain | difference, level of detection per cell, gain / loss / below detection / no data, sensor notes |
| `harmonise(a, b, out_a, out_b)` | pulse densities | both surveys thinned to a common pulse density, as tiles |
| `canopy_gaps`, `gap_change` | CHMs | gaps with outlines; formation and closure; size distributions |
| `tree_change(trees_a, trees_b, chm)` | trees of `als.find_trees` | per-tree height growth with its level of detection; survivors, damaged, dead, recruits |
| `metric_change(a, b)` | `als.grid_metrics` | metric differences with a level of detection per cell |
| `pai_change(profile_a, profile_b)` | `als.gap_profile` | PAI and PAD change with their uncertainty |

## A worked example

[`synthetic.als_epochs`](../api/synthetic.md) flies a synthetic forest
twice with known changes: trees felled singly, eight felled together to
open a gap, the trees of the west half grown by 1 m, a second sensor flown
higher and sparser, and the second survey delivered displaced by
`(0.3, -0.2, 0.15)` m. Two gabled buildings east of the forest give roofs
of four aspects.

```python
from sylva import als, change, synthetic

ep = synthetic.als_epochs(n_trees=230, gap=(70.0, 70.0, 8), seed=1)
a, b = ep.write_tiles("2019", "2024", size=50.0, epsg=32755)    # 370,785 and 204,161 returns

al = change.align_surveys(a, b, block_size=60.0, stable_classes=(2, 6))
chm = change.chm_change(a, b, resolution=2.0, alignment=al)
gaps = change.gap_change(chm, height=2.0, min_area=10.0, years=5)

ha, hb = change.harmonise(a, b, "2019_h", "2024_h")               # a common pulse density
kw = dict(resolution=1.0, window=als.LinearWindow(0.0, 0.2, 3.0, 20.0), max_cr=20, hmin=5.0)
trees = change.tree_change(als.find_trees(ha, **kw), als.find_trees(hb, **kw),
                           change.chm_change(a, b, 1.0, alignment=al, harmonise=True),
                           alignment=al)
metrics = change.metric_change(a, b, 20.0, metrics=["zq95", "cover"], alignment=al,
                               harmonise=True)
print(al.report(), chm.report(), gaps.report(), trees.report(years=5), sep="\n")
```

```text
ALS alignment (field): constant offset dx +0.296 ± 0.023, dy -0.198 ± 0.015, dz +0.150 ± 0.001 m
  blocks   4 of 6 of 60 m with an estimate; dx fixed by the data in 1, dy in 4
  spread   0.023 m before, 0.020 m after (median over blocks, robust)
  Birge    1.00, 1.00, 1.00 (x, y, z)
  noise    0.020 m (a), 0.020 m (b)
CHM change at 2 m: 14,400 m² assessed, 240 m² no data
  gain                    3,808 m²  (26.4 %)
  loss                      400 m²  (2.8 %)
  below_detection        10,192 m²  (70.8 %)
  level of detection  median 0.57 m (5 to 95 %: 0.02 to 2.05)
  volume  +4,338 / -4,297 m³ over the significant cells
  survey a  12.6 pulses/m², 2.06 returns per pulse, mean |scan angle| 14.3 deg
  survey b  5.6 pulses/m², 2.56 returns per pulse, mean |scan angle| 13.8 deg
  note: pulse density differs: 12.6 pulses/m² in survey a and 5.6 in survey b ...
  note: returns per pulse differ (2.06 in survey a, 2.56 in survey b; ...
Gaps (at most 2 m high, at least 10 m²): 10 in survey a (27.6 %), 9 in survey b (29.0 %)
  formed   200 m² (gaps new: 1, expanded: 1)
  ...
```

The offset is recovered to 4, 2 and 0.1 mm, each within its standard
deviation; the ground alone (a slope of one aspect in x) would have fixed
`dy` but left `dx` to its prior, and `dz` with it (see below). The gap formed
covers 200 m², as the true canopy does; the cells that crossed the 2 m
threshold without a significant change (120 m²) are `uncertain`. The
survivors grew by +0.62 ± 0.06 m on average (true +0.56 m), though only 12
of 86 individually beyond their level of detection of about 1 m. The
example runs in about 10 s.

## Aligning the surveys

Two surveys are never quite in one frame: GNSS/INS trajectories drift by
centimetres to decimetres between flights, and a vertical offset of 10 cm
over a hectare is 1,000 m³ of spurious change in a DTM difference.
`align_surveys` estimates the offsets of the second survey on surfaces that
do not change: ground (class 2) and, with `stable_classes=(2, 6, 11)`,
roofs and roads.

Every stable return of the second survey (one per `sample_spacing` square)
is compared with a plane fitted to the first survey's stable returns within
`radius` of it. A survey displaced by `(dx, dy, dz)` stands
`dz - gx dx - gy dy` above a plane of gradient `(gx, gy)`, so the vertical
offset comes from every sample and the horizontal ones from the variety of
slopes and aspects, as in the co-registration of elevation models of Nuth
and Kääb (2011). Each block of `block_size` m is solved by iteratively
reweighted least squares (Huber 1964 weights), refitting the planes at the
moved positions until the offsets settle; samples within
`correlation_length` of each other count as one independent sample, since
neighbouring planes share returns and surveys err in patches.

Two things keep the uncertainty honest:

- **Noisy gradients.** A plane fitted to a few returns has a noisy
  gradient, and noise in a regressor both pretends to fix a horizontal
  offset and biases it towards zero. The moment matrix is corrected by the
  gradients' own covariance (the correction for errors in variables;
  Fuller 1987), and a direction whose remaining information is within the
  sampling fluctuation of that correction is given none.
- **Undetermined directions.** On flat ground, or on a slope of one
  aspect, a horizontal shift along the slope looks exactly like a vertical
  one. A Gaussian prior of `horizontal_prior` (0.3 m) keeps such a
  horizontal offset near zero, its standard deviation stays near the
  prior, and the vertical offset absorbs the part the data fix
  (`dz - gx dx`), with an uncertainty that says so. The report counts the
  blocks whose data fixed each horizontal offset.

The blocks are then combined (`model`): `"constant"` pools the information
of all blocks into one offset, `"blocks"` keeps each block's own (blocks
without enough samples take the constant one), and `"field"` (the default)
pools the blocks at each block centre with Gaussian weights of `smoothing`
m and interpolates bilinearly between centres, for surveys whose offset
drifts along a flight. Either way the uncertainty is widened by the Birge
ratio of the blocks about the model, so that blocks disagreeing more than
their uncertainties allow widen the uncertainty rather than being averaged
away. `al.table()` and `al.to_csv()` give each block's estimate, its
standard deviation and its residual from the model; `al.offset_at(x, y)`,
`al.sigma_at(x, y)` and `al.apply(cloud)` use the model, and
`ALSAlignment.constant(offset, sigma)` states a known one. Blocks are
chunked whole, so each estimate uses exactly the returns in and around its
block whatever `chunk_size` and `workers` are.

## Surface change

`chm_change(a, b, resolution)` (and `dtm_change`, or `surface_change` with
`surface="dsm"`) grids both surveys in one pass with one algorithm, the
highest return per cell, each normalised by a DTM of its own ground returns
(`dtm_method="plane"`: at each DTM cell a plane fitted to the ground
returns around it, with its standard error). `subcircle` replaces each
return by eight points on a circle, and
`first_returns=True` grids first returns only.

**Sampling.** The highest of a cell's returns is a sample of its canopy: it
falls short of the top by an amount that depends on the number of pulses
and on how the canopy's heights are spread in the cell, and at a crown's
edge one survey may hit the crown where the other hits only ground. Both
effects are read from the data. Within a cell each pulse contributes its
highest return there. If the canopy did not change, the pulses of the two
surveys are samples of one surface, and every split of the pooled values
into the two surveys' numbers of pulses is equally likely. The distribution
of `max(b) - max(a)` over those splits follows exactly from the order
statistics of the pooled values (if the highest value falls in the first
survey, with probability `n_a / N`, the second survey's highest is `v_j`
with probability `C(j - 1, n_b - 1) / C(N - 1, n_b)`, and the other way
round): a permutation test on the maxima (Pitman 1937). Its mean is the
change sampling alone gives (`bias`: negative where the second survey is
sparser), and its central `confidence` interval bounds the change expected
without any.

**Other errors.** Each survey's DTM under the cell errs by the standard
error of its ground plane plus `interpolation_error` times the distance to
the nearest ground return; the alignment errs horizontally (its standard
deviation times the surface's gradient) and, for DSMs and DTMs, vertically
(a CHM is normalised by each survey's own ground, which cancels a vertical
offset). The permutation interval is widened by these, taken as normal and
independent of the sampling, to a half-width `lod = sqrt(h² + (z s)²)`,
with `h` its sampling half-width and `s` their combined standard deviation.
A change above the interval `[lower, upper]` is a `gain`, below it a
`loss`, within it `below_detection`; a cell with fewer than `min_returns`
pulses in either survey is `no_data`. The DoD of the significant cells
(`chm.dod`: volumes and areas, as [`change.dod`](change.md#rasters-of-difference)
gives them) and the rasters (`chm.write("out/")`) follow.

**Choosing the resolution.** The test needs pulses: a cell with 3 pulses
of the sparser survey cannot tell whether it missed a sparse crown by
chance. On the synthetic surveys (12.5 and 5.5 pulses/m²), 87 % of the
cells whose canopy fell by more than 3 m were a significant loss at 2 m,
and 64 % at 1 m. A cell should hold about 20 pulses of the sparser survey.

**Scan patterns.** The permutation assumes that a survey's pulses fall at
random places in a cell. Surveys are planned with pulses about as far
apart along the scan as along the track, and the test then holds: with the
same sensor flown twice over an unchanged canopy, 2.6 to 4 % of the cells
were called changed at 95 % confidence. A pattern with pulses 15 times
farther apart across the track than along it (the default `scan_rate` of
`synthetic.als_flight` at 4,000 pulses/s) puts a cell's pulses in one or two
lines, and 11 % of the cells were then called changed: the pulses are not
independent samples of the cell.

### Sensor differences and harmonisation

A difference is change only if both surveys measured the canopy the same
way (Næsset 2009; Disney et al. 2010). The permutation covers the pulse
density, which is the largest effect on a CHM: the highest of fewer returns
falls shorter of the top, so a sparser second survey reads as canopy loss.
`chm.bias` is that expected change per cell and `chm.median_bias` its
median over the canopy. Other differences are not sampling and are not
covered. A wider footprint or a more sensitive receiver triggers first
returns higher in a crown and records more returns per pulse; a different
scan angle sees crown sides and gaps differently. `chm.sensors` compares
the surveys (pulse density, returns per pulse, share of single-return
pulses, mean absolute scan angle, most returns per pulse) and `chm.notes`
names every difference large enough to bias the comparison.

`harmonise=True` thins the denser survey, pulse by pulse, to the other's
pulse density in each `density_cell` square: in a square where it has more
first returns, each of its pulses is kept with probability
`n_other / n_own`, chosen by a hash of the pulse (GPS time and flight line)
and `seed`, so every return of a pulse shares its fate and the choice does
not depend on the chunks. `change.harmonise(a, b, out_a, out_b)` writes the
same thinning as tiles, for tree detection or anything else run on both
surveys. On the synthetic surveys the median CHM difference over unchanged
canopy was -0.07 m without harmonisation (the density effect) and +0.04 m
with it (what remains is the second sensor's wider footprint); harmonising
costs detection power, since the denser survey loses pulses.

## Gaps

`canopy_gaps(chm, height, min_area)` finds the gaps of one CHM as
ForestGapR's `getForestGaps` does (Silva et al. 2019): connected cells
(across edges and corners, `connectivity=8`) no higher than `height`, with
an area between `min_area` and `max_area`. The default 2 m follows Brokaw's
(1982) definition of a gap as an opening reaching down to within 2 m of
the ground. Each gap has its area, centre, heights and outline along cell
edges (a polygon per part, with holes; `to_geojson`), and
`size_exponent()` fits a power law to the sizes by maximum likelihood
(Clauset et al. 2009), as gap sizes often follow one (Fisher et al. 2008;
Asner et al. 2013).

`gap_change(chm_change)` finds the gaps of both surveys and classes every
cell: `formed` (in a gap of the second survey only, with a significant
loss), `closed` (in a gap of the first only, with a significant gain),
`stable_gap`, `canopy`, `no_data`, and `uncertain` where a cell crossed the
threshold without a significant change. This is ForestGapR's
`GapChangeDec` with a level of detection: without one, every cell whose
canopy moved across the threshold by noise alone would count. Each gap of
the second survey is `new`, `expanded`, `stable` or `uncertain`, each of the
first `closed`, `shrunk`, `stable` or `uncertain`, with its area formed or
closed; `summary()` gives gap fractions, areas, annual formation and
closure rates and the size exponents.

## Trees

`tree_change(trees_a, trees_b, chm)` takes the trees of
[`als.find_trees`](als_trees.md) in each survey and the CHM change of the
same surveys (at the resolution the trees were found at). The trees are
matched by the optimal assignment of
[`change.match_trees`](change.md#matching-trees) (Kuhn 1955; Munkres 1957),
the tree height standing in for the diameter: a pair must lie within
`max_distance` (1.5 m) and its height may grow by at most `max_growth`
(30 %) or fall by at most `max_drop` (20 %) of the larger height. The CHM
change says what became of the rest:

| Status | Survey | When |
|---|---|---|
| `survivor` | both | matched |
| `damaged` | both | matched, with a significant loss over `damage_fraction` (30 %) of the crown or a height fall beyond its level of detection; or unmatched, with a tree of the second survey standing on significantly lowered canopy within `2 max_distance` of its top (a broken top) |
| `dead` | first | unmatched, with a significant loss over `dead_fraction` (50 %) of its crown |
| `recruit` | second | unmatched, its top on a significant gain from below `1 - max_growth` of its height |
| `released` | second | unmatched, its top on a significant loss: an understorey tree exposed by a neighbour's fall |
| `undetected` | either | unmatched, without such a change: the canopy is still there, but no tree was found on it in the other survey |
| `unobserved` | either | less than `min_observed` (50 %) of the crown has data in both surveys |

A height is the value of the top's CHM cell, and its standard deviation
that cell's in the CHM change (`sigma_a`, `sigma_b`: the spread of its
highest pulse under the permutation, and the DTM's error); a height change
beyond `1.96` times their combination is `growth` or `decrease`, otherwise
`below_detection`. `summary(area, years)` gives the counts, the mean height
growth of the survivors with its standard error (from their scatter, and
from their measurement uncertainty alone), the crown area of the dead and
lost from the damaged, and annual mortality and recruitment rates (Sheil
et al. 1995); `grid(resolution)` the same per cell, `to_csv` and
`to_pandas` the table.

Which trees can be compared depends on tree detection, which misses
overtopped trees and splits crowns at low pulse densities (see
[Airborne trees](als_trees.md#validation)). A top found in one survey only
is `undetected`, not a death or a recruit; expect many of them where
detection is poor, and compare the counts with that in mind. Detect the
trees of both surveys at a common pulse density (`change.harmonise`) so
that they are found alike.

## Area-based metrics and plant area index

`metric_change(a, b, resolution, metrics)` computes the metrics of
[`als.grid_metrics`](als_metrics.md) for both surveys in one pass (heights
above each survey's own DTM, the second moved by `alignment`, optionally
harmonised) and tests each difference as the CHM is tested: if nothing
changed, the pulses of both surveys that fell in one `stratum` m square (2
m) sample the same canopy, and any reassignment of them between the
surveys that keeps each survey's number of pulses in each square is as
likely as the observed one. The mean and central interval of
`metric(b) - metric(a)` over `permutations` (100) reassignments, drawn from
a stream seeded by the cell and `seed`, give `bias`, `[lower, upper]` and
the class. Keeping pulses in their squares keeps the pattern in which each
survey sampled the cell, which a resampling of the cell's pulses as
independent draws would ignore.

The test covers sampling only. With thousands of returns per cell, the
sensor differences above are significant: on the synthetic surveys the
second sensor's wider footprint raised the mean height of unchanged cells
by 0.7 m and their cover by 3 %, and harmonising the pulse density does not
remove it. Metrics compared across sensors measure the sensors as well as
the forest.

`pai_change(profile_a, profile_b)` compares two
[`als.gap_profile`](als_canopy.md) results on one grid and set of layers
(give both the same `bounds`, `resolution`, `bin_size`, `min_height` and
`max_height`), each made with its own survey's trajectory. With `W` pulses
in a cell, a share `P` reaching the ground and a mean extinction `k`,
`PAI = -ln P / k` (MacArthur and Horn 1969), and its standard deviation
from the binomial variance of `P` is `sqrt((1 - P) / (W P)) / k`. A cell no
pulse crossed is `saturated`: its PAI is only a lower bound. `profile` and
`profile_change(profile_a, profile_b, mask)` compare the pooled plant area
density profile of an area layer by layer in the same way. The profiles are
not moved by an alignment: use cells much larger than the horizontal
offset.

## Chunks and workers

Every result is computed cell by cell from the returns near the cell, in a
fixed order, with interpolation weights taken from the position on the
whole catalogue's grid, so the number of workers never changes it, and
neither, in practice, does the chunk size: across tile-sized chunks and 17,
23, 30 and 45 m chunks on one to four workers, the alignment blocks, the
classes and nearly every value were identical, and the few values that
differed (a handful of cells) did so in the last bits. The buffer is raised
automatically to hold whole cells and the ground planes of the DTM under
them (up to 16 m); with `harmonise`, also a `density_cell`.

On 36 ha (the synthetic pair copied 5 by 5: 9.3 and 5.1 million returns in
36 and 48 tiles of 100 m), on eight threads, the alignment took 7.5 s, the
CHM change 8 s (harmonised, 8 s), the metric change of three metrics with
100 permutations per cell 41 s, and writing the harmonised tiles 14 s.

## Command line

```bash
sylva als-change 2019/ 2024/ change/ --align --stable-classes 2,6 --harmonise \
    --resolution 2 --gaps --gap-min-area 10 --years 5
sylva als-tree-change 2019/ 2024/ trees.csv --align --harmonise --resolution 1 \
    --window-linear 0 0.2 3 20 --hmin 5 --grid totals/ --grid-resolution 50 --years 5
```

`als-change` writes `<surface>_a`, `_b`, `_difference`, `_lod` and
`_classes` rasters (`--format asc` or `tif`), the alignment's blocks
(`alignment.csv`) and, with `--gaps`, the gaps of both surveys with their
status as GeoJSON, and prints the reports. `als-tree-change` finds the
trees of both surveys (on harmonised copies with `--harmonise`), compares
them and writes one CSV row per tree, and with `--grid` the per-cell totals
as rasters. Both take `--pattern`, `--chunk-size`, `--buffer` and
`--workers`.

## Validation

`synthetic.als_epochs` (100 m of forest, 100 trees of 12 to 25 m with
crowns of closed outline, two buildings) was flown twice per seed, with the
changes of the worked example: 4 trees felled singly, 5 felled to open a
gap, the trees of the west half grown by 1.0 m, the second survey
displaced by `(0.3, -0.2, 0.15)` m. Three sets: the default sensors
(12.5 and 5.5 pulses/m², the second flown 50 % higher; 5 seeds), the same
sensor twice (5 seeds), and a closed canopy (230 trees, 8 felled for the
gap; 3 seeds). "Unchanged" cells lie more than 1.5 m beyond the crown of
every changed tree; the truth of a cell is the highest point of the scene
above it.

**Alignment** (`model="constant"`, 60 m blocks):

| Stable surfaces | Error dx, dy, dz (mean abs.) | Standard deviation | Largest error / sd |
|---|---|---|---|
| ground | 0.29, 0.003, 0.014 m | 0.30, 0.019, 0.015 m | 1.0 |
| ground and roofs | 0.005, 0.002, 0.000 m | 0.022, 0.015, 0.001 m | 0.7 |

The terrain slopes uniformly in x, so ground alone leaves `dx` to the prior
and says so; its `dz` absorbs the shift along the slope, within its
standard deviation. The roofs fix all three.

**CHM change** (95 % confidence, aligned unless stated):

| Sensors | Cell | Harmonised | False change, unchanged canopy | Unchanged ground | Canopy fallen > 3 m: loss | Grown 1 m: gain | Median difference, unchanged canopy |
|---|---|---|---|---|---|---|---|
| different | 2 m | no | 3.1 % | 0.4 % | 87 % | 65 % | -0.07 m |
| different | 2 m | yes | 3.0 % | 0.3 % | 85 % | 52 % | +0.04 m |
| different | 2 m, not aligned | no | 6.8 % | 0.3 % | 87 % | 67 % | -0.08 m |
| different | 1 m | no | 1.9 % | 0.9 % | 64 % | 25 % | -0.09 m |
| same | 2 m | no | 4.0 % | 0.7 % | 90 % | 68 % | 0.00 m |
| same | 1 m | no | 2.6 % | 1.2 % | 84 % | 26 % | 0.00 m |
| closed canopy | 2 m | no | 3.8 % | 0.4 % | 96 % | 74 % | -0.07 m |

Without the alignment, the 0.3 m horizontal offset doubles the false
changes at crown edges; the DTM difference is then a significant rise over
99 % of the area, and aligned 0.05 % (median |difference| 1.6 mm).

**Gaps** (closed canopy, 2 m cells): the gap cells formed covered 200,
364 and 204 m² in the true canopy; 90, 91 and 84 % of them were `formed`,
with 16 to 20 m² formed elsewhere, and 88 to 176 m² `uncertain`. The gap
at the felled trees was `new` or `expanded` in every seed, and the gap
fractions (27 to 29 %) were within 2 points of the true ones.

**Trees** (`dalponte2016` at 1 m, window a fifth of the height; a
detected top is a true tree's when within 1.5 m of its highest point):

| Sensors | Trees found in the first survey | Felled to open the gap: `dead` | Felled singly: `dead` | Height change error, grown (mean, RMSE) | Unchanged (mean, RMSE) | 1 m growth beyond its LoD | Unchanged below detection | False `dead`, false `recruit` |
|---|---|---|---|---|---|---|---|---|
| different | 492 | 14 of 16 | 7 of 14 | -0.07, 0.17 m | -0.07, 0.15 m | 20 % (LoD 1.04 m) | 99 % | 0, 0 |
| different, harmonised | 776 | 10 of 14 | 4 of 14 | +0.04, 0.19 m | +0.02, 0.17 m | 26 % (LoD 1.18 m) | 99 % | 0, 2 |
| same | 492 | 16 of 16 | 11 of 14 | +0.00, 0.11 m | +0.01, 0.10 m | 65 % (LoD 0.72 m) | 100 % | 0, 0 |
| closed canopy, different | 390 | 8 of 9 | 0 of 4 | -0.06, 0.16 m | -0.07, 0.13 m | 17 % (LoD 1.13 m) | 100 % | 0, 0 |

In every set at least 98 % of the height changes lay within their level
of detection of the truth. Harmonising removes the density
bias of the height changes (-0.07 m, the sparser survey's tops falling
shorter) at the cost of more false tops in the thinned survey and so fewer
trees assessed; the felled trees not called `dead` were `undetected`
(their crowns overlapped standing neighbours, so less than half of the
crown lost canopy), never `survivor`. Of the 1,300 or so tops found in
the second survey only (over all 26 comparisons, mostly splits of large
crowns), 3 were called `recruit` and the rest `undetected`; no tree grew
in.

**Metrics** (20 m cells, 100 permutations, 2 m strata): with the same
sensor, 12 % (zq95), 3 % (zmean) and 6 % (cover) of the unchanged cells
were called changed. With the different sensors, 45 to 70 %: the second
sensor's footprint raised zmean by 0.7 m and cover by 3 %, a real
difference in what the sensors measure, which harmonising the density does
not remove.

`tests/test_change_als.py` repeats these checks on a smaller pair and
covers the rest: harmonisation in the pass equals harmonised tiles, results
across chunks and workers, analytic gap outlines with holes, PAI change
from counts and from two flights, and invalid input.

## Limitations

- Offsets only: rotations between surveys are modelled as a smoothly
  varying offset field, which suits trajectory drift but not a rotated
  delivery frame.
- The levels of detection cover sampling, DTMs and alignment, not sensor
  differences; the notes name them, and harmonisation removes only the
  density effect.
- A cell needs pulses: resolve at about 20 pulses of the sparser survey
  per cell. Strongly anisotropic scan patterns make the tests
  anti-conservative.
- Tree-level results inherit the detection of `als.find_trees`; unmatched
  tops are labelled `undetected` rather than counted as deaths or recruits.
- Gaps, trees and PAI work on the rasters and profiles of the whole area in
  memory (as `als.chm` returns them); the point passes are chunked.
- PAI profiles are not aligned; they must share their grid and layers.

## References

Asner, G. P., Kellner, J. R., Kennedy-Bowdoin, T., Knapp, D. E., Anderson, C., & Martin,
R. E. (2013). Forest canopy gap distributions in the southern Peruvian Amazon. *PLoS
ONE*, *8*(4), e60875. <https://doi.org/10.1371/journal.pone.0060875>

Brokaw, N. V. L. (1982). The definition of treefall gap and its effect on measures of
forest dynamics. *Biotropica*, *14*(2), 158–160. <https://doi.org/10.2307/2387750>

Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in
empirical data. *SIAM Review*, *51*(4), 661–703. <https://doi.org/10.1137/070710111>

Disney, M. I., Kalogirou, V., Lewis, P., Prieto-Blanco, A., Hancock, S., & Pfeifer, M.
(2010). Simulating the impact of discrete-return lidar system and survey
characteristics over young conifer and broadleaf forests. *Remote Sensing of
Environment*, *114*(7), 1546–1560. <https://doi.org/10.1016/j.rse.2010.02.009>

Fisher, J. I., Hurtt, G. C., Thomas, R. Q., & Chambers, J. Q. (2008). Clustered
disturbances lead to bias in large-scale estimates based on forest sample plots.
*Ecology Letters*, *11*(6), 554–563. <https://doi.org/10.1111/j.1461-0248.2008.01169.x>

Fuller, W. A. (1987). *Measurement error models*. Wiley.
<https://doi.org/10.1002/9780470316665>

Huber, P. J. (1964). Robust estimation of a location parameter. *The Annals of
Mathematical Statistics*, *35*(1), 73–101. <https://doi.org/10.1214/aoms/1177703732>

Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research
Logistics Quarterly*, *2*(1–2), 83–97. <https://doi.org/10.1002/nav.3800020109>

MacArthur, R. H., & Horn, H. S. (1969). Foliage profile by vertical measurements.
*Ecology*, *50*(5), 802–804. <https://doi.org/10.2307/1933693>

Munkres, J. (1957). Algorithms for the assignment and transportation problems.
*Journal of the Society for Industrial and Applied Mathematics*, *5*(1), 32–38.
<https://doi.org/10.1137/0105003>

Næsset, E. (2009). Effects of different sensors, flying altitudes, and pulse
repetition frequencies on forest canopy metrics and biophysical stand properties
derived from small-footprint airborne laser data. *Remote Sensing of Environment*,
*113*(1), 148–159. <https://doi.org/10.1016/j.rse.2008.09.001>

Nuth, C., & Kääb, A. (2011). Co-registration and bias corrections of satellite
elevation data sets for quantifying glacier thickness change. *The Cryosphere*,
*5*(1), 271–290. <https://doi.org/10.5194/tc-5-271-2011>

Pitman, E. J. G. (1937). Significance tests which may be applied to samples from any
populations. *Supplement to the Journal of the Royal Statistical Society*, *4*(1),
119–130. <https://doi.org/10.2307/2984124>

Sheil, D., Burslem, D. F. R. P., & Alder, D. (1995). The interpretation and
misinterpretation of mortality rate measures. *Journal of Ecology*, *83*(2), 331–333.
<https://doi.org/10.2307/2261571>

Silva, C. A., Valbuena, R., Pinagé, E. R., Mohan, M., de Almeida, D. R. A., North
Broadbent, E., Jaafar, W. S. W. M., de Almeida Papa, D., Cardil, A., & Klauberg, C.
(2019). ForestGapR: An R package for forest gap analysis from canopy height models.
*Methods in Ecology and Evolution*, *10*(8), 1347–1356.
<https://doi.org/10.1111/2041-210X.13211>
