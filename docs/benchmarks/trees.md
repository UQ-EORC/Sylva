# Tree detection and segmentation

`benchmarks/evaluate_trees.py` scores the detect → segment → prune pipeline
against manually segmented reference plots (per-point tree labels) as
instance segmentation: a reference tree counts as found when its
best-overlapping sylva tree has IoU ≥ 0.5; predicted trees outside the
reference coverage are ignored. `benchmarks/evaluate_external.py` scores
raycloudtools' `rayextract trees` output the same way.

| Site (points) | sylva F1 | raycloudtools F1 | sylva TP / ref | sylva mean IoU |
|---|---|---|---|---|
| LITCH_CHERLET (8 M) | 0.96 | 0.97 | 124 / 127 | 0.98 |
| ROBSON_CHERLET (9 M) | 0.57 | 0.55 | 91 / 149 | 0.73 |
| WYTHAM_CHERLET (14 M) | 0.77 | 0.73 | 125 / 180 | 0.85 |
| OFENTAL_CHERLET (15 M) | 0.57 (0.63 with `min_quality_short=0.15`) | 0.73 | 56 / 89 | 0.69 |

(`benchmarks/eval_v3_cherlet.log`.) Run-to-run noise is about ±0.02 F1. The
remaining losses are crown leakage between interlocking neighbours (about
16 % of each reference tree's points end up on an adjacent tree at Robson
and Ofental) and, at Ofental, extra candidates from low conifer branches;
`prune_trees(min_quality_short=0.15)` trades a little rainforest recall for
that.
