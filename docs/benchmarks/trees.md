# Tree detection and segmentation

The detect → segment → prune pipeline is scored
against manually segmented reference plots (per-point tree labels) as
instance segmentation: a reference tree counts as found when its
best-overlapping sylva tree has IoU ≥ 0.5; predicted trees outside the
reference coverage are ignored. raycloudtools' `rayextract trees` output is
scored the same way.

| Site (points) | sylva F1 | raycloudtools F1 | sylva TP / ref | sylva mean IoU |
|---|---|---|---|---|
| LITCH_CHERLET (8 M) | 0.96 | 0.97 | 124 / 127 | 0.98 |
| ROBSON_CHERLET (9 M) | 0.57 | 0.55 | 91 / 149 | 0.73 |
| WYTHAM_CHERLET (14 M) | 0.77 | 0.73 | 125 / 180 | 0.85 |
| OFENTAL_CHERLET (15 M) | 0.57 (0.63 with `min_quality_short=0.15`) | 0.73 | 56 / 89 | 0.69 |

(Those figures are the per-site runs recorded earlier. On the CHERLET *test*
split of two of them, re-scored with the current defaults, the loss is now
mostly where two stems stand within a metre of each other.)

| test split | trees found / ref | was | mean IoU | recall | precision |
|---|---|---|---|---|---|
| Litchfield (savanna) | 136 / 145 | 135 | 0.97 | 0.99 | 0.98 |
| Wytham (temperate broadleaf) | 156 / 232 | 130 | 0.87 | 0.95 | 0.92 |
| Robson Creek (rainforest) | 117 / 222 | 83 | 0.69 | 0.86 | 0.79 |
| Ofental (conifer) | 72 / 117 | 63 | 0.72 | 0.86 | 0.82 |

**Seeds used to be too wide.** A tree starts from the points within
`seed_radius` of its stem at `seed_height`; at 0.5 m those discs overlap
wherever stems are close, and the stronger seed takes the other's crown. Of
the 102 Wytham trees missed before, 79 had been swallowed by a neighbour
whose stem stood a median of **0.3 m away** — coppice stools and low forks,
not distant trees. Halving `seed_radius` to 0.25 m and `merge_radius` to
0.2 m finds 22 more trees (130 → 152), lifts mean IoU 0.83 → 0.85 and
precision 0.88 → 0.90, and costs Litchfield nothing (135 → 136).

A **sparser graph and a steeper cost** help on top of that: `k` 10 → 6 and
`power` 3 → 4 (the exponent on edge length, so a path pays more for a long
hop between two crowns). Found trees: Litchfield 136, Wytham 152 → 154,
Robson 105 → 110, Ofental 63 → 68, with fewer spurious trees everywhere.

A **finer graph** pays for itself where the stand is dense: the cloud is
thinned to `voxel_size` before the graph is built, and 0.05 → 0.03 m gives
Wytham 154 → 156, Robson 110 → 117 and Ofental 68 → 72, each with *fewer*
false trees and a better IoU, for about 1.7× the time. Litchfield does not
move at any resolution. 0.02 m gains two more at Wytham for 2.5× the time,
which did not seem worth making the default.

Other things tried, none of which helped: a denser graph (`k=20`, Wytham
136), gravity 0.3 (140), wood costs (142), dropping the height prior (139 at
Wytham, 61 at Ofental), a wider height-prior radius (62 at Ofental), no angle
penalty (48 at Ofental, much worse), `max_edge` 0.5 (no change), and no
pruning at all (142) — so the limit is not the candidate list.

**Where the remaining loss is.** Robson and Ofental are still the hard ones,
and for different reasons. At Ofental 41 of the 54 missed trees are absorbed
by one neighbour, and the missed trees are short (median 10.1 m) where the
matched ones are tall (16.4 m): suppressed conifers under dominants, whose
stems *are* detected (81 % have a candidate within 0.5 m, median 0.05 m) but
whose crowns are taken by the tree above. Robson keeps 222 reference trees
against 655 candidates, so it over-detects in the understorey as well as
merging. Both point at the cost model rather than at detection or pruning.

Of Robson's 212 spurious trees, **123 sit almost entirely (> 80 % of their
points) on what the reference calls ground** — rainforest understorey, median
6.2 m tall — and only 40 are fragments of a labelled tree. The same tension
as the savanna shrubs: real vegetation the benchmark does not count as a
tree. `prune_trees(min_quality_short=0.15)` trims about a quarter of them for
two found trees.

Tried on Ofental and rejected, in case it saves someone the experiment:
softening the height prior, which is what hands a dominant its suppressed
neighbour's crown (`height_prior_power` 1 → 0.75 → 0.5 → 0.25 gives 68 → 68 →
66 → 52, and the same knob costs Wytham trees below 0.5), turning the
understorey competition off (67) or down to 5 m (68), and seeding at 3 m
instead of 1.5 m (65, and 40 more false trees).

Run-to-run noise is about ±0.02 F1. The
remaining losses are crown leakage between interlocking neighbours (about
16 % of each reference tree's points end up on an adjacent tree at Robson
and Ofental) and, at Ofental, extra candidates from low conifer branches;
`prune_trees(min_quality_short=0.15)` trades a little rainforest recall for
that.
