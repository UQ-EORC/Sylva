# Sylva

NOTE: This software is still in ALPHA development.

Terrestrial (TLS) and airborne (ALS) laser scanning processing for forest
ecology and remote sensing. A Rust core (the `sylva-rs` crate) does the work; Python gets a
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
| [`sylva.io`](api/io.md) | LAS/LAZ (typed extra bytes, CRS), PLY (incl. raycloudtools ray clouds), XYZ/CSV/PTS, RIEGL `.rxp` via RiVLib, RiSCAN project parsing |
| [`sylva.coords`](api/coords.md) | translation, rotation, recentring, reprojection between CRSs, applying transform files to many scans |
| [`sylva.filters`](api/filters.md) | voxel / random / Poisson-disk subsampling, box & cylinder crops, statistical & radius outlier removal, PCA normals, planarity, Euclidean clustering, kNN |
| [`sylva.ground`](api/ground.md) | Cloth Simulation Filter and Progressive Morphological Filter ground classification, DTM (lowest point, TIN, natural neighbour or IDW), height normalisation, CHM |
| [`sylva.interpolate`](api/interpolate.md) | attributes carried between clouds, grids from points (IDW, TIN, natural neighbour), rasters sampled onto points |
| [`sylva.masks`](api/masks.md) | masks and crops by polygons, rasters, attribute expressions and distance to another cloud |
| [`sylva.trees`](api/trees.md) | RANSAC circle fitting, stem detection & DBH, basal area, taper profiles, graph-based tree segmentation, tree heights, crown metrics |
| [`sylva.canopy`](api/canopy.md) | voxel grids, contact-frequency PAD profiles, zenith-ring gap fraction, hinge/Miller LAI, ray-traced density grids from pulse data |
| [`sylva.voxels`](api/voxels.md) | AMAPVox-style ray-traced voxels (port of raycloudtools `rayvoxel`): echo-weighted free / potential path lengths, FPL / PPL / transmittance / Bailey attenuation, analytic or estimated leaf-angle `G`, PAD / LAD / WAD, occlusion, sub-voxel exploration, QSM wood volume, `.vox` export |
| [`sylva.registration`](api/registration.md) | Kabsch, point-to-point / point-to-plane (trimmed) ICP, scan merging |
| [`sylva.coreg`](api/coreg.md) | marker-free coregistration of scan positions from the trees and the ground, with reflectors where present, and a report of the stem agreement |
| [`sylva.qsm`](api/qsm.md) | cylinder fitting, geodesic skeletonisation, cylinder QSMs with volumes and branch orders, a QSM for every tree of a plot, buttress meshes, `_trees.txt` export |
| [`sylva.change`](api/change.md) | change between two epochs of a plot: alignment, tree matching and increments, plot summaries, point change (C2C, M3C2, DEM of difference, voxel occupancy), QSM change, each labelled trusted or not |
| [`sylva.als`](api/als.md) | airborne lidar over tiled areas: catalogues and buffered chunks, ground, DTM, CHM, normalisation, filtering, retiling and thinning |
| [`sylva.als_metrics`](api/als_metrics.md) | area-based metrics (the lidR standard set, cover, gap fraction) as rasters or plot tables |
| [`sylva.als_trees`](api/als_trees.md) | tree tops, crowns (watershed, Dalponte 2016, Li 2012), crown outlines and labelled tiles |
| [`sylva.als_canopy`](api/als_canopy.md) | pulses from the flight trajectory, gap-fraction and PAD profiles corrected for beam angle, ray-traced voxels |
| [`sylva.fusion`](api/fusion.md) | TLS and ALS together: a plot registered on a survey, stems linked to airborne trees, merged clouds and plant area profiles, plot values upscaled with leave-one-out checks |
| [`sylva.waveform`](api/waveform.md) | LAS wave packets and PulseWaves, Gaussian decomposition into echoes, waveforms to pulses |
| [`sylva.Shots`](api/shots.md) | pulse-centric data (origin, direction, CSR echoes) for ray-based metrics, with a compact Parquet file format that stores misses without far points and streams into the voxeliser |
| [`sylva.leaves`](api/leaves.md) | graph-based leaf/wood separation, leaf angle distributions, leaf area density from points or voxels, leaf meshes placed on a QSM |
| [`sylva.quality`](api/quality.md) | scan noise and per-scan registration offsets measured on tree stems |
| [`sylva.synthetic`](api/synthetic.md) | small synthetic trees, plots, scans, repeat surveys, airborne flights and waveforms with known answers, for examples and tests |

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
- [Coordinates](guide/coordinates.md), [interpolation](guide/interpolation.md)
  and [masking](guide/masking.md): moving clouds between frames and
  coordinate systems, carrying values between clouds and grids, and selecting
  points by area, raster, expression or proximity.
- [Change detection](guide/change.md): two epochs of a plot, from alignment
  and matched trees to point, voxel and QSM change, with what can and cannot
  be trusted.
- Airborne lidar: [tiles](guide/als.md) (ground, DTM, CHM over large areas),
  [area-based metrics](guide/als_metrics.md), [trees](guide/als_trees.md),
  [canopy structure from the pulses](guide/als_canopy.md) and
  [full waveforms](guide/waveform.md).
- [Example notebooks](examples/index.md), one per stage, on a tile of the
  TERN Litchfield plot and on synthetic data.
- [Benchmarks](benchmarks/trees.md) against reference plots, felled trees and
  raycloudtools.
- The [function index](api/index.md) lists every public function; the API
  reference pages hold each one's parameters with units, return values and
  exceptions, generated from the docstrings.
- [Command line](guide/cli.md): the `sylva` command for shell pipelines.

## Related tools

- [Segfix](https://github.com/tim-devereux/segfix): a GUI for correcting the
  instance segmentation of a plot; it reads and writes the `tree_id` column
  Sylva produces ([how](guide/trees.md#correcting-labels-by-hand)).
- [raycloudtools](https://github.com/csiro-robotics/raycloudtools): ray
  clouds, which `sylva.read` and `Shots.from_ray_cloud` accept.
- Everything Sylva ports or takes influence from is cited on the [references](references.md) page.

## Author and licence

Tim Devereux, The University of Queensland.

Free software under the GNU General Public License v3.0 or later: you may
use, study, change and share it, and anything you distribute that builds on
it carries the same licence. See the LICENSE file in the repository.

Cite it as: Devereux, T. (2026). *Sylva: terrestrial and airborne laser
scanning processing for forest ecology* (version 0.2.1) [Computer software].
The University of Queensland. <https://github.com/UQ-EORC/Sylva>

```bibtex
@software{sylva,
  author       = {Devereux, Tim},
  title        = {Sylva: terrestrial and airborne laser scanning processing
                  for forest ecology},
  year         = {2026},
  version      = {0.2.1},
  organization = {The University of Queensland},
  url          = {https://github.com/UQ-EORC/Sylva}
}
```
