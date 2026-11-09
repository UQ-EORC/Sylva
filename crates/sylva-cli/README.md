# sylva-cli

The `sylva` command: LiDAR processing for forest ecology and remote sensing
research, built on [`sylva-rs`](https://crates.io/crates/sylva-rs).

```bash
cargo install sylva-cli

sylva info plot.laz
sylva ground plot.laz plot_norm.laz --dtm dtm.asc
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva qsm tree.ply tree_qsm.csv
sylva shots rays.laz plot.parquet
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2
```

The binary covers the single-cloud commands (`info`, `convert`, `ground`,
`trees`, `chm`, `pad`, `shots`, `voxel`, `qsm`); the Python package's `sylva`
command adds coregistration, tiles, airborne lidar, change and fusion.
Documentation: <https://uq-eorc.github.io/Sylva/> (the [command line
guide](https://uq-eorc.github.io/Sylva/guide/cli/) describes every command
of the Python entry point; the binary's options can differ in detail, see its
`--help`).
Licence: GPL-3.0-or-later. To cite Sylva, see the
[citation](https://github.com/UQ-EORC/Sylva#citation) in the repository.
