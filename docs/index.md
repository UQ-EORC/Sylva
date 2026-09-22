# Sylva

Terrestrial laser scanning (TLS) processing for forest ecology and remote
sensing. A Rust core (the `sylva-rs` crate) does the work; Python gets a
numpy-friendly API (`import sylva`) and a `sylva` command.

```bash
pip install maturin && maturin develop --release
```

```python
import sylva
from sylva import ground, trees

cloud = ground.classify_ground_csf(sylva.read("plot.laz"))
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, resolution=0.5))
stems = trees.detect_stems(cloud)                  # positions and DBH
labels = trees.segment_trees(cloud, stems)         # tree_id per point
```

## What is in it

| Module | What it does |
|---|---|
| [`sylva.io`](api/io.md) | LAS/LAZ (typed extra bytes), PLY (incl. raycloudtools ray clouds), XYZ/CSV/PTS, RIEGL `.rxp` via RiVLib, RiSCAN project parsing |
| [`sylva.filters`](api/filters.md) | voxel / random / Poisson-disk subsampling, box & cylinder crops, statistical & radius outlier removal, PCA normals, planarity, Euclidean clustering, kNN |
| [`sylva.ground`](api/ground.md) | Cloth Simulation Filter and Progressive Morphological Filter ground classification, DTM, height normalisation, CHM |
| [`sylva.trees`](api/trees.md) | RANSAC circle fitting, stem detection & DBH, taper profiles, graph-based tree segmentation, tree heights, crown metrics |
| [`sylva.canopy`](api/canopy.md) | voxel grids, contact-frequency PAD profiles, zenith-ring gap fraction, hinge/Miller LAI, ray-traced density grids from pulse data |
| [`sylva.voxels`](api/voxels.md) | AMAPVox-style ray-traced voxels (port of raycloudtools `rayvoxel`): echo-weighted free / potential path lengths, FPL / PPL / transmittance / Bailey attenuation, analytic or estimated leaf-angle `G`, PAD / LAD / WAD, occlusion, sub-voxel exploration, QSM wood volume, `.vox` export |
| [`sylva.registration`](api/registration.md) | Kabsch, point-to-point / point-to-plane (trimmed) ICP, scan merging |
| [`sylva.qsm`](api/qsm.md) | cylinder fitting, geodesic skeletonisation, cylinder QSMs with volumes and branch orders, `_trees.txt` export |
| [`sylva.Shots`](api/shots.md) | pulse-centric data (origin, direction, CSR echoes) for ray-based metrics, with a compact Parquet file format that stores misses without far points and streams into the voxeliser |
| [`sylva.leaves`](api/leaves.md) | graph-based leaf/wood separation, leaf angle distributions, leaf area density from points or voxels, leaf meshes placed on a QSM |
| [`sylva.quality`](api/quality.md) | scan noise and per-scan registration offsets measured on tree stems |
| [`sylva.synthetic`](api/synthetic.md) | small synthetic trees, plots and scans with known answers, for examples and tests |

## Where to start

- [Install](install.md), then the [plot workflow](guide/quickstart.md): ground,
  stems, segmentation, crowns.
- [Point clouds and files](guide/io.md),
  [filtering and registration](guide/preprocessing.md) and
  [trees and crowns](guide/trees.md): each stage and its settings.
- [Operational use](guide/operational.md): batch processing, quality flags,
  reproducibility and what has been validated.
- [Pulse data](guide/pulses.md): `Shots`, RiSCAN projects, gap fraction, and
  the Parquet shots file format.
- [Ray-traced voxels](guide/voxels.md): AMAPVox-style attenuation and plant
  area density.
- [QSMs](guide/qsm.md): wood filtering and cylinder models of single trees.
- [Example notebooks](examples/index.md), one per stage, on synthetic data.
- [Benchmarks](benchmarks/trees.md) against reference plots, felled trees and
  raycloudtools.
- The [function index](api/index.md) lists every public function; the API
  reference pages hold each one's parameters with units, return values and
  exceptions, generated from the docstrings.
- [Command line](guide/cli.md): the `sylva` command for shell pipelines.

## Related tools

- [Segfix](https://github.com/tim-devereux/segfix) — a GUI for correcting the
  instance segmentation of a plot; it reads and writes the `tree_id` column
  Sylva produces ([how](guide/trees.md#correcting-labels-by-hand)).
- [raycloudtools](https://github.com/csiro-robotics/raycloudtools) — ray
  clouds, which `sylva.read` and `Shots.from_ray_cloud` accept.
- Everything Sylva implements is cited on the [references](references.md) page.
