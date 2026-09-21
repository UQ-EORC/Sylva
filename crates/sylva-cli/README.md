# sylva-cli

The `sylva` command: terrestrial laser scanning processing for forest ecology,
built on [`sylva-rs`](https://crates.io/crates/sylva-rs).

```bash
cargo install sylva-cli

sylva info plot.laz
sylva ground plot.laz plot_norm.laz --dtm dtm.asc
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva qsm tree.ply tree_qsm.csv
sylva shots rays.laz plot.parquet
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2
```

See the [Sylva repository](https://github.com/tim-devereux/Sylva) for the
documentation. Licence: MIT.
