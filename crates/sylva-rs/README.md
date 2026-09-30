# sylva-rs

[![crates.io](https://img.shields.io/crates/v/sylva-rs)](https://crates.io/crates/sylva-rs)
[![PyPI](https://img.shields.io/pypi/v/sylva-rs)](https://pypi.org/project/sylva-rs/)

Terrestrial and airborne laser scanning processing for forest ecology: the Rust
core of [Sylva](https://github.com/UQ-EORC/Sylva), which also ships as the
`sylva-rs` Python package (`pip install sylva-rs`, then `import sylva`) and the
`sylva` command (`cargo install sylva-cli`).

```bash
cargo add sylva-rs
```

- `PointCloud` with typed attributes; LAS/LAZ, PLY, text and RIEGL `.rxp` I/O;
  coordinate transforms and reprojection
- filters, k-d tree queries, clustering, PCA normals, interpolation, masks
- ground classification (CSF, PMF), DTM, CHM
- stem detection, DBH, tree segmentation, crown metrics
- registration (Kabsch, ICP) and marker-free coregistration of scan positions
- cylinder QSMs, leaf / wood separation and leaves on QSMs
- `Shots`: pulse data with misses, stored as Parquet and streamed
- ray-traced voxel grids with FPL / PPL / transmittance attenuation and plant,
  leaf and wood area density
- airborne lidar over tiled areas: metrics, trees, canopy from the trajectory,
  full waveforms; TLS and ALS fusion; change detection between surveys

```rust
use sylva_rs::{ground, io, trees};

let cloud = io::read("plot.laz")?;
let mask = ground::csf_ground_mask(&cloud.xyz, &ground::CsfParams::default());
let ground_pts: Vec<_> = cloud.xyz.iter().zip(&mask).filter(|(_, &g)| g).map(|(p, _)| *p).collect();
let dtm = ground::make_dtm(&ground_pts, 0.5, None)?;
let heights = ground::heights_above(&cloud.xyz, &dtm);
let stems = trees::detect_stems(&cloud.xyz, &heights, &trees::StemParams::default());
```

Documentation: <https://uq-eorc.github.io/Sylva/>. Licence: GPL-3.0-or-later.
To cite Sylva, see the
[citation](https://github.com/UQ-EORC/Sylva#citation) in the repository.
