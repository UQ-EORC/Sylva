# sylva-rs

Terrestrial laser scanning (TLS) processing for forest ecology: the Rust core
of [Sylva](https://github.com/tim-devereux/Sylva), which also ships as the
`sylva-rs` Python package (`import sylva`) and the `sylva` command
(`cargo install sylva-cli`).

- `PointCloud` with typed attributes; LAS/LAZ, PLY, text and RIEGL `.rxp` I/O
- filters, k-d tree queries, clustering, PCA normals
- ground classification (CSF, PMF), DTM, CHM
- stem detection, DBH, tree segmentation, crown metrics
- registration (Kabsch, point-to-point / point-to-plane ICP)
- cylinder QSMs with leaf / wood separation
- `Shots`: pulse data with misses, stored as Parquet and streamed
- `voxel`: ray-traced voxel grids with FPL / PPL / transmittance attenuation
  and plant, leaf and wood area density

```rust
use sylva_rs::{ground, io, trees};

let cloud = io::read("plot.laz")?;
let mask = ground::csf_ground_mask(&cloud.xyz, &ground::CsfParams::default());
let ground_pts: Vec<_> = cloud.xyz.iter().zip(&mask).filter(|(_, &g)| g).map(|(p, _)| *p).collect();
let dtm = ground::make_dtm(&ground_pts, 0.5, None)?;
let heights = ground::heights_above(&cloud.xyz, &dtm);
let stems = trees::detect_stems(&cloud.xyz, &heights, &trees::StemParams::default());
```

Licence: MIT.
