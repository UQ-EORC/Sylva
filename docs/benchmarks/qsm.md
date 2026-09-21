# QSM benchmarks

## Destructive harvest

Single-tree QSMs are scored against felled-tree
volumes: 72 trees from Momo Takoudjou 2017, Gonzalez de Tanago 2017 and Burt
2021, with wood density assigned per tree and agreement statistics computed
the same way for every method:

| method | volume bias | rRMSE | CCC | DBH bias / rRMSE |
|---|---|---|---|---|
| rayextract | −11.8 % | 48.6 % | 0.920 | −23 % / 29 % |
| sylva | −3.8 % | 19.9 % | 0.989 | +2 % / 13 % |

Study
biases are Momo −5 %, GdtM −3 %, Burt +1 %. The remaining scatter is a handful of
co-dominant Peruvian trees whose second limb is barely sampled: the pipe
model has to guess it. Never cap or anchor stem radii on a breast-height
slice: at 1.3 m the big tropical trees are 3 m-wide buttress stars where a
circle explains 15 % of the points.

## CHERLET plots against raycloudtools

On the CHERLET plots, per matched tree, sylva / raycloudtools volume medians
are Wytham 0.78 (0.86 for trees over 20 cm DBH; plot totals 126 vs 126 m³),
Ofental 0.69 (totals 74 vs 60 m³), Litchfield 0.38 (0.85 over 20 cm; totals
17 vs 22 m³) and Robson 0.19. Against the earlier chain-prior model the
plots gained ten times as many cylinders per tree and branch orders 4–5
instead of 1–3. QSM DBH agrees with the detected DBH to within ~15 % on
the temperate plots; on the rainforest saplings the wood filter's medium
step lowers it (Litchfield 0.72, Robson 0.49 against 0.83 / 0.62 with
`medium_threshold=1.0`). The big Robson trees that sylva models at a
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
