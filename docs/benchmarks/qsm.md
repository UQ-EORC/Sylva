# QSM benchmarks

## Destructive harvest

Single-tree QSMs are scored against felled-tree
volumes: the 72 trees that every compared method reconstructed, from [Momo Takoudjou et al. (2018)](../references.md), [Gonzalez de
Tanago et al. (2018)](../references.md) and [Burt et al. (2021)](../references.md), with wood density assigned per tree and agreement statistics computed
the same way for every method; rayextract is raycloudtools' tree
reconstruction ([Devereux et al. 2026](../references.md)):

| method | volume bias | rRMSE | CCC | DBH bias / rRMSE |
|---|---|---|---|---|
| rayextract | −11.8 % | 48.6 % | 0.920 | −23 % / 29 % |
| sylva | +0.5 % | 16.8 % | 0.992 | −1 % / 13 % |

The sylva row is the present code: by study, −1.9 % / 16.2 % on the 40
Cameroon trees (Momo Takoudjou), +4.6 % / 18.7 % on the 28 of Peru, Guyana
and Indonesia (Gonzalez de Tanago) and −3.9 % / 6.6 % on the 4 of Burt et
al. The median absolute log ratio of modelled to harvested volume is 0.104.
With the clouds thinned to 3 and 5 cm the volume rRMSE is 15.0 % and 19.4 %
([QSM guide](../guide/qsm.md#how-fine-does-the-cloud-have-to-be)).

### Trees that played no part in any choice

The rule for unmeasured radii (a cross-section area in proportion to the
length of branch carried, the width of a section's points as a ceiling, and
the test of sections that have no taper to be judged by;
[QSM guide](../guide/qsm.md#what-is-not-measured)) was chosen on simulated
trees and on the clouds of 34 felled eucalypts
([Calders et al. 2015](../references.md)), and the 72 trees above were scored
afterwards. Two further sets were scored once, after every choice was made:
65 trees of four species felled in Belgium, scanned leaf-off
([Demol et al. 2021](../references.md)), and 26 felled in Guyana
([Lau et al. 2019](../references.md)). The middle rows are the code as it
stood before that rule, with a radius in proportion to length; the rayextract
volumes are from the authors' own runs of raycloudtools on the same clouds.

| trees | method | volume bias | rRMSE | CCC | median abs. log ratio |
|---|---|---|---|---|---|
| Belgium, 65 | rayextract | −9.2 % | 15.8 % | 0.959 | 0.097 |
| | radius by length | −6.7 % | 22.4 % | 0.923 | 0.135 |
| | sylva | −5.1 % | 11.2 % | 0.981 | 0.076 |
| Guyana, 26 | rayextract | +6.3 % | 38.1 % | 0.944 | 0.104 |
| | radius by length | +6.7 % | 53.0 % | 0.903 | 0.130 |
| | sylva | +9.5 % | 31.5 % | 0.962 | 0.080 |
| Eucalypts, 34 (used in development) | rayextract | −29.4 % | 41.8 % | 0.825 | 0.296 |
| | radius by length | −17.5 % | 27.2 % | 0.933 | 0.184 |
| | sylva | −7.6 % | 18.4 % | 0.970 | 0.111 |

On the 72 large tropical trees the radius-by-length code scored −1.9 % /
16.9 %, as well as the present one. It did so by two errors that cancel in
trees whose volume is mostly trunk and limbs: side branches filled in at a
fifth to a half of their radius, and limbs near large forks and in foliage
filled in too thick. The smaller and finer trees of the table show the
difference.

Against the 20 simulated trees with exact wood meshes (99 leaf-off scans),
which share nothing with Sylva's own tree generator, the median volume
error is −3.2 % (radius by length: −10.1 %; rayextract on the same scans
−10.4 %). The 69 scans of trees
over 1 m³ give −3.0 % and 8.0 % rRMSE (−9.5 % and 11.0 %). Trees under 1 m³
read a median 13 % low (34 %) with a wide spread (50 %): one tree of
0.14 m³ reads twice its volume.

How the exponent was settled: on the simulated trees and on those meshes the
true radius of unmeasured wood, relative to the nearest measured section
towards the root, goes as the 0.44th to 0.52nd power of the length of branch
carried, whether that is counted as length, as tips or as points; a radius
in proportion to the longest path with an area share, the earlier rule,
gives a median 0.13 to 0.45 of the true radius. On scanned trees the same
can be asked of measured sections alone, by predicting each from a measured
section at least a metre nearer the root: the square root gives a median
0.92 to 1.00 of the measured radius on the eucalypts, the 72 tropical trees
and the simulated scans, the proportional rule 0.59 to 0.85.

Earlier versions in the same run: 2.1.0 scores −2.7 % / 18.4 % / 0.991 and
2.0.0 +0.1 % / 25.6 % / 0.982, and 1.1.0, before the cylinder stage and the
wood filter were rewritten, −2.0 % / 24.1 % / 0.984. **Those three figures
flatter the models.** Up to 2.1.0 a node without an accepted circle sat at
the mean of its points, and the skeleton began at the lowest patch of
points near the base, so the stems of large trees stepped sideways and back
and ran along the ground before turning up: the largest carried two to
four times their stem's real length in the lowest metre. That excess stood
in for the crown limbs, which are seen from below over less than half their
ring and were not measured at all. With the nodes placed as they are now
and the limbs still unmeasured (`min_arc_deg=180`), the same trees score
−18 % / 41 %; measuring the limbs from the arcs that were seen brings them
to the row above.

Against the points themselves, on 71 of the trees (the cylinders of 5 cm
radius and more, each against the points within its length): the surface
lies a median 1.2 to 1.6 cm from the points (2.1.0: 2.4 to 4.4 cm), the
axis 1.9 cm from the centre of the ring of points (2.6 to 5.2 cm),
and consecutive cylinders of 20 cm radius and more turn a median 0.6°
(6.5°). Of the turns between cylinders of 5 cm and more, 11 % exceed 20°
(17.5 %).

The settings were chosen on simulated trees and the eucalypts before these
trees were scored. On the 72 trees `noise_ratio` from 1.2 to 2 gives 16.5 to
17.8 % rRMSE and `min_snr` of 4 and 6 give 17.0 and 18.3 %, with the bias
moving from +3 % to −3 %; `width_slack` of 1 and 2 give 17.4 and 16.8 %
(18.5 % without the ceiling) and `pipe_excess` of 2 and 4 give 17.1 and
17.2 % (18.8 % without the test). The remaining scatter
is mostly in co-dominant Peruvian trees whose second limb is barely
sampled, where the radius of an unmeasured limb has to be inferred from
the length it carries. DBH is less well read than in 2.1.0 (13 % rRMSE
against 12 %): on a buttressed base the lowest sections are circles around
the flanges, and the stem below them takes their radius. For single trees,
never cap or anchor stem radii on
a breast-height slice: at 1.3 m the big tropical trees are 3 m-wide
buttress stars where a circle explains 15 % of the points. (On plots the
cap and the stem table are what keep unanchored leafy stems to size; see
the [QSM guide](../guide/qsm.md).)

Fitting on other wood inputs (graph-based labels, and all points weighted by
a wood confidence) was scored the same way with version 1.1.0 on the 73 trees with a cloud at
hand (one more than above: that tree is left out of the method comparison
because rayextract did not reconstruct it); none beat
the passage filter (22.0 % rRMSE on that set, against 22.6-26.6 %); the
table is in the [QSM guide](../guide/qsm.md#wood-labels-and-wood-weights).

## CHERLET plots against raycloudtools

On the CHERLET plots, per matched tree, sylva / raycloudtools volume medians
(measured with version 1.0.0, before the cylinder stage and the wood filter
were replaced) are Wytham 0.78 (0.86 for trees over 20 cm DBH; plot totals 126 vs 126 m³),
Ofental 0.69 (totals 74 vs 60 m³), Litchfield 0.38 (0.85 over 20 cm; totals
17 vs 22 m³) and Robson 0.19. Against the earlier chain-prior model the
plots gained ten times as many cylinders per tree and branch orders 4–5
instead of 1–3. QSM DBH agrees with the detected DBH to within ~15 % on
the temperate plots; on the rainforest saplings the wood filter of that
version lowered it (Litchfield 0.72, Robson 0.49). The big Robson trees that sylva models at a
tenth of rayextract's volume are detection failures (a 0.08 m stem
candidate on a 0.8 m tree) and truncated segments, not QSM ones. Segmentation
leaves points below 0.5 m unassigned unless they sit within 1 m (1.5 DBH)
of their tree's base (`segment_trees(low_height=, low_radius=)`): the
graph otherwise reaches ground remnants, litter and understorey along the
surface and the QSM models them as fat horizontal cylinders (a Wytham oak
lost 0.4 of 2.7 m³ that way); the rule does not change the detection
scores. The Robson gap is the single-scan rainforest crowns and trunks:
where a trunk is sparsely sampled the anisotropy wood filter keeps few of
its points, the stem goes unmeasured and its radius comes from the prior.
