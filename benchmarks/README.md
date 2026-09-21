# Benchmarks

The scripts behind the numbers on the documentation's benchmark pages
([tree detection](../docs/benchmarks/trees.md), [QSMs](../docs/benchmarks/qsm.md)).
They are not part of the package and are not installed with it. The data
cannot be redistributed here; each section says where it comes from.

Run them from the repository root with Sylva installed (`maturin develop
--release`), plus `pandas` and `matplotlib` for the harvest and render scripts.
Data locations come from two environment variables (see `paths.py`):

| Variable | Default | Holds |
|---|---|---|
| `SYLVA_DATA` | `~/data` | one folder per plot |
| `SYLVA_HARVEST` | `~/data/harvest_benchmark` | the destructive-harvest benchmark (`code/harvest`, `data/harvest`, `results/harvest`) |

Sylva writes its results to `<site>/pytls/` and to files named `*_pytls*`.
The names predate the project's renaming and are kept so that existing
results stay readable.

## Tree detection and segmentation

Reference: manually segmented TLS plots with a tree label per point
(`<site>/reference/<site>_reference_no_ground.npy`), here the four CHERLET
plots (Litchfield, Robson Creek, Wytham Woods, Ofental).

| Script | Does |
|---|---|
| `evaluate_trees.py SITE...` | ground → stems → segmentation → pruning on each plot, scored as instance segmentation (a reference tree is found when its best-overlapping tree has IoU ≥ 0.5); writes `<site>/pytls/segmented.laz` and `trees.csv` |
| `evaluate_external.py SITE...` | scores raycloudtools `rayextract trees` output (`<site>/raycloudtools_*/trees/tree_*.ply`) the same way |

`eval_v3_cherlet.log` is the run the published table was taken from.

## QSMs

| Script | Does |
|---|---|
| `run_qsms.py SITE...` | a QSM for every tree of a segmented plot, plus `plot_qsm.ply`, one mesh for the plot coloured per tree |
| `compare_qsm_rct.py SITE...` | per matched tree, Sylva volume and DBH against raycloudtools' `_trees.txt` |
| `render_qsm.py SITE TREE_ID [out.png]` | draws one QSM over its points, for inspection |

### Destructive harvest

Reference: 72 felled and weighed trees from Momo Takoudjou et al. 2017,
Gonzalez de Tanago et al. 2017 and Burt et al. 2021. The benchmark folder
supplies the tree clouds, the reference table, results of other methods, and
the density assignment and statistics (`benchmark_ref.py`).

| Script | Does |
|---|---|
| `harvest_qsm.py [--mesh]` | builds a QSM for each reference tree cloud; writes `out_pytls_ref/<cloud>.csv` and `pytls_metrics.csv` |
| `harvest_benchmark.py` | volume, DBH and height agreement (R², CCC, bias, rRMSE) beside every other method the benchmark holds results for |
| `harvest_structure.py [--other NAME=DIR]` | reads each method's models into one cylinder representation and compares stem, branch and taper structure |
