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
| sylva | +0.1 % | 25.6 % | 0.982 | −3 % / 12 % |

The sylva row is the present code. Version 1.1.0, before the cylinder
stage and the wood filter were rewritten, scores −2.0 % / 24.1 % / 0.984
and +2 % / 12 % in the same run, and the median absolute log ratio of
modelled to harvested volume is 0.121 against 0.123. The remaining scatter
is mostly in co-dominant Peruvian trees whose second limb is barely
sampled, where the radius of an unmeasured limb has to be inferred from
the length it carries. For single trees, never cap or anchor stem radii on
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
