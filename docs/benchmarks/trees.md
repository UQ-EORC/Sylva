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

| test split | trees found / ref | mean IoU | recall | precision |
|---|---|---|---|---|
| Litchfield | 136 / 145 | 0.96 | 0.99 | 0.97 |
| Wytham | 152 / 232 | 0.85 | 0.94 | 0.90 |

**Seeds used to be too wide.** A tree starts from the points within
`seed_radius` of its stem at `seed_height`; at 0.5 m those discs overlap
wherever stems are close, and the stronger seed takes the other's crown. Of
the 102 Wytham trees missed before, 79 had been swallowed by a neighbour
whose stem stood a median of **0.3 m away** — coppice stools and low forks,
not distant trees. Halving `seed_radius` to 0.25 m and `merge_radius` to
0.2 m finds 22 more trees (130 → 152), lifts mean IoU 0.83 → 0.85 and
precision 0.88 → 0.90, and costs Litchfield nothing (135 → 136).

Other things tried on Wytham, none of which helped: a denser graph (`k=20`,
136), gravity 0.3 (140), wood costs (142), dropping the height prior (139),
and no pruning at all (142) — so the limit is not the candidate list.

Run-to-run noise is about ±0.02 F1. The
remaining losses are crown leakage between interlocking neighbours (about
16 % of each reference tree's points end up on an adjacent tree at Robson
and Ofental) and, at Ofental, extra candidates from low conifer branches;
`prune_trees(min_quality_short=0.15)` trades a little rainforest recall for
that.
