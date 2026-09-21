# Command line

Installing the Python package provides a `sylva` command; `crates/sylva-cli`
builds a Rust binary with the same subcommands and no Python dependency
(`cargo install --path crates/sylva-cli`). `sylva <command> --help` lists the
options of each.

```bash
sylva info plot.laz
sylva ground plot.laz plot_norm.laz --dtm dtm.asc
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva chm plot_norm.laz chm.asc
sylva pad plot_norm.laz --voxel 0.5
sylva shots rays.laz plot.parquet                  # ray cloud -> shots file
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2 --laser VZ-400 --attenuation fpl ppl
sylva qsm tree.ply tree_qsm.csv
```
