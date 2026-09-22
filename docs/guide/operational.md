# Operational use

Sylva is written to serve two purposes: a library to experiment with and a
tool that processes plots in bulk. This page covers the second: which steps
to chain, how to run them unattended, which quality flags to check, and what
has been validated.

## The standard chain

| Stage | Functions | Output |
|---|---|---|
| Read and register | `read_riscan_project`, `ScanPosition.read`, `registration.merge_scans` | project-frame cloud with `scan_id` |
| Check the scan | `quality.stem_noise(...).summary()` | noise and per-scan registration |
| Thin and clean | `filters.voxel_downsample`, `filters.statistical_outlier_removal` | working cloud |
| Ground | `ground.classify_ground_csf`, `make_dtm`, `normalize_height`, `make_chm` | `classification`, `height`, DTM, CHM |
| Trees | `trees.detect_stems`, `merge_branches`, `segment_trees`, `tree_heights`, `prune_trees`, `crown_metrics_all` | tree table, `tree_id` per point |
| Wood models | `qsm.wood_points`, `qsm.build_qsm`, `QSM.metrics` | cylinders and tree architecture per tree |
| Canopy (pulses) | `canopy.GapProfile`, `voxels.ray_voxelize` | PAI, PAVD profiles, clumping, voxel PAD |
| Coverage | `RayVoxelGrid.occlusion_profile`, `voxels.tree_sampling` | what was and was not seen |

Every stage takes the output of the one before, and nothing is changed in
place, so a pipeline can keep any intermediate result it needs.

Where the tree labels have to be right, they can be corrected between the
tree and wood-model stages in [Segfix](https://github.com/tim-devereux/segfix),
which reads and writes the `tree_id` column in place; see
[Trees and crowns](trees.md#correcting-labels-by-hand).

## A batch script

```python
"""Process every plot under data/, one output folder per plot."""
import json, logging, sys, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import sylva
from sylva import filters, ground, quality, trees, qsm

log = logging.getLogger("plots")
PARAMS = {"voxel": 0.01, "dtm_res": 0.5, "min_height": 3.0}


def process(src: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cloud = filters.voxel_downsample(sylva.read(src), PARAMS["voxel"])
    cloud = ground.classify_ground_csf(cloud)
    dtm = ground.make_dtm(cloud, PARAMS["dtm_res"])
    cloud = ground.normalize_height(cloud, dtm)
    dtm.to_ascii_grid(out / "dtm.asc")

    stems, _ = trees.merge_branches(cloud, trees.detect_stems(cloud))
    labels = trees.segment_trees(cloud, stems)
    trees.tree_heights(cloud, labels, stems, percentile=99)
    stems, labels = trees.prune_trees(stems, labels, min_height=PARAMS["min_height"])
    crowns = trees.crown_metrics_all(cloud, labels)

    rows = []
    for t in stems:
        model = qsm.build_qsm(qsm.wood_points(cloud[labels == t.tree_id]), base_xy=(t.x, t.y))
        model.to_csv(out / f"qsm_{t.tree_id:04d}.csv")
        m = model.metrics()
        rows.append({**t.as_dict(), **crowns.get(t.tree_id, {}),
                     "qsm_volume": m["total_volume"], "measured_volume_fraction": m["measured_volume_fraction"]})
    pd.DataFrame(rows).to_csv(out / "trees.csv", index=False)
    sylva.write(cloud.with_attrs(tree_id=labels.astype(np.int32)), out / "segmented.laz")
    return {"n_trees": len(stems), "quality": quality.stem_noise(cloud, stems=stems).summary()}


for src in sorted(Path("data").glob("*.laz")):
    out = Path("results") / src.stem
    if (out / "run.json").exists():
        continue                                    # resumable: skip finished plots
    try:
        result = process(src, out)
        manifest = {"input": str(src), "sylva": sylva.__version__, "params": PARAMS, **result}
        (out / "run.json").write_text(json.dumps(manifest, indent=1, default=float))
    except Exception:
        log.error("%s failed:\n%s", src, traceback.format_exc())
```

- **Manifest.** Each plot writes `run.json` last, holding the input, the
  Sylva version and the parameters. A plot with no `run.json` either failed
  or is unfinished, so a rerun picks it up.
- **Parallelism.** Sylva's Rust core runs in parallel on all cores and
  releases the Python GIL. Run plots one after another, or cap the threads
  per process with `RAYON_NUM_THREADS=8` when running several at once.
- **Memory.** It is set by the largest step: a full-resolution plot cloud,
  or a ray-traced voxel grid (about 0.25 kB per voxel while tracing). Thin
  first. Stream pulse data from a shots file, since `ray_voxelize` accepts
  the path and reads it a few row groups at a time.

## The command line

For shell pipelines and workflow managers (Make, Snakemake, Nextflow),
each stage is also a command. Commands exit with status 1 and a one-line
message on error.

```bash
sylva ground plot.laz plot_norm.laz --dtm dtm.tif
sylva trees plot_norm.laz -o trees.csv --segment segmented.laz
sylva chm plot_norm.laz chm.tif
```

See [Command line](cli.md) for every command and option.

## Reproducibility

- **Seeds.** Every function with a random element takes a `seed` (default
  0), including RANSAC fits, random thinning and the azimuths of
  reconstructed misses.
- **Repeat runs.** With the same inputs, parameters and version, stem
  detection, segmentation, QSMs and ray-traced voxels repeat exactly; this
  is checked in the test suite. Results can change between Sylva versions,
  so record `sylva.__version__` with every output and pin the version for a
  campaign (`pip install sylva-rs==X.Y.Z`).
- **Parameters.** Keep the parameters you pass in the manifest; defaults can
  change between versions.

## Quality flags to check

Sylva reports how far to trust each result. Carry these flags into the
output tables and filter on them:

| Result | Flag | Look for |
|---|---|---|
| Scan | `stem_noise(...).summary()` | `sigma_local` (range noise), `registration_rms`, `worst_scan` |
| Stem | `Tree.quality`, `Tree.inlier_fraction`, `Tree.n_slices` | low values: a partly seen or doubtful stem |
| Tree height | `tree_sampling(...)["above_observed_fraction"]` | low: the top may be hidden and the tree taller |
| QSM | `metrics()["measured_volume_fraction"]` | low: most of the volume came from priors |
| PAI (gap profile) | `report()["saturated"]` | True: PAI is a lower bound set by the pulse count |
| Voxels | `occlusion_profile()["total"]` | large `unobserved` or `occluded` shares |
| Ray clouds | `UserWarning` from `Shots.from_ray_cloud` | no sensor positions: ray-traced products are invalid |

## What has been validated

| Output | Reference | Result | Page |
|---|---|---|---|
| Tree detection and segmentation | 4 manually segmented plots | F1 0.57–0.96 | [Trees](../benchmarks/trees.md) |
| QSM volume | 72 destructively harvested trees | bias −3.8 %, rRMSE 19.9 %, CCC 0.989 | [QSMs](../benchmarks/qsm.md) |
| Tree metrics | 20 simulated trees with exact meshes | height −2 %, DBH −1 %, crown area −3 %, branch zenith 3° | [QSMs](../benchmarks/qsm.md) |
| Leaf/wood labels | 30 manually labelled tropical trees | accuracy 0.90, mIoU 0.79 | [Leaves](leaves.md) |
| PAI (gap profile) | pylidar on TERN plots; hemispherical photos | within 4 % where unsaturated; r = 0.96 | [Canopy](../benchmarks/canopy.md) |
| Stem noise | split-half agreement, synthetic stems | 1.1 mm between halves | [Scan quality](quality.md) |

Not validated against independent data: vertical PAD profile shape, DTM
accuracy, and leaf area from points (a lower bound wherever foliage was
occluded).

## Errors

| Exception | Raised for |
|---|---|
| `OSError` | files that are missing, unreadable or malformed |
| `ValueError` | invalid parameters, unknown methods, too few points, RiVLib not found |
| `KeyError` | a named attribute that the data lacks |
| `ImportError` | optional dependencies (`rasterio` for GeoTIFF) |

A plot with no detectable stems returns an empty list, not an error, so check
counts before indexing.
