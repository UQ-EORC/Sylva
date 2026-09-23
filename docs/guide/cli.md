# Command line

Installing the Python package provides a `sylva` command. Each command reads
its input, writes its outputs and prints a one-line summary. On error it
prints `sylva: error: ...` to stderr and exits with status 1, so commands can
be chained in shell scripts and workflow managers. `sylva <command> --help`
shows every option with its default, and `sylva --version` prints the
version.

```bash
sylva info plot.laz
sylva ground plot.laz plot_norm.laz --dtm dtm.tif
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva chm plot_norm.laz chm.tif
sylva pad plot_norm.laz --voxel 0.5 > pad.csv
sylva shots rays.laz plot.parquet                  # ray cloud -> shots file
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2 --laser VZ-400 --attenuation fpl ppl
sylva qsm tree.ply tree_qsm.csv
sylva qsm-plot plot_trees.laz trees.csv --cylinders qsms/
```

`crates/sylva-cli` builds a standalone Rust binary with the same commands
and no Python dependency (`cargo install --path crates/sylva-cli`). Its
options can differ in detail, so check its `--help`.

## Commands

### `info`

`sylva info INPUT`: prints the point count, bounds, and each attribute's type
and range.

### `convert`

`sylva convert INPUT OUTPUT [--voxel SIZE]`: converts between any readable
and writable formats (by extension). `--voxel` keeps one point per voxel of
that size (m).

### `ground`

`sylva ground INPUT OUTPUT [--method csf|pmf] [--resolution 0.5] [--dtm PATH]`

Classifies ground (`classification` 2), builds a DTM at `--resolution` (m),
and writes the cloud with a `height` attribute. `--dtm` also writes the
terrain (`.tif` needs `rasterio`, otherwise `.asc`).

### `trees`

`sylva trees INPUT [-o CSV] [--min-dbh 0.05] [--min-height 3.0] [--segment PATH]`

Runs detection, segmentation, heights, pruning and crown metrics on a
height-normalised cloud and writes one CSV row per tree: `tree_id`, `x`,
`y`, `dbh`, `height`, quality columns and the crown metrics. The CSV goes to
stdout without `-o`. `--segment` writes the cloud with a `tree_id`
attribute.

### `chm`

`sylva chm INPUT OUTPUT [--resolution 0.5]`: writes the canopy height model
of a height-normalised cloud and prints the canopy cover above 2 m.

### `pad`

`sylva pad INPUT [--voxel 0.5]`: prints a `height,pad` CSV of the
contact-frequency profile on stdout, and the PAI on stderr.

### `qsm`

`sylva qsm INPUT OUTPUT [--bin-length 0.3]`: builds a cylinder model of one
tree (ideally its wood points), writes the cylinders as CSV and prints the
summary.

### `qsm-plot`

`sylva qsm-plot INPUT OUTPUT [options]`: builds a QSM for every tree of a
segmented cloud (one that carries a `tree_id` attribute, as `sylva trees
--segment` writes) and puts a row per tree in `OUTPUT`.

| Option | Default | What |
|---|---|---|
| `--tree-attr NAME` | `tree_id` | attribute holding the tree id |
| `--cylinders DIR` | none | also write `tree<id>.csv` cylinders per tree |
| `--voxel M` | 0.01 | thin each tree to this spacing first |
| `--bin-length M` | 0.1 | geodesic shell width |
| `--min-points N` | 2000 | skip trees with fewer points |
| `--buttress` | off | mesh a flanged base and count it in the volume |
| `--no-wood` | off | skip the leaf/wood filter (the cloud is wood already) |

Trees that are too small or that cannot be fitted are listed on stdout rather
than stopping the run.

### `shots`

`sylva shots INPUT OUTPUT [--double]`: converts a raycloudtools ray cloud
(`sx,sy,sz` or `nx,ny,nz` attributes) to a shots file.

### `voxel`

`sylva voxel INPUT OUTPUT [options]`: ray-traced voxel grid, written as
AMAPVox `.vox` or a `.txt` table. `INPUT` is a shots file (streamed, so
memory stays small) or a ray cloud.

| Option | Default | Meaning |
|---|---|---|
| `--voxel` | 0.1 | voxel size (m) |
| `--bounds X0 Y0 Z0 X1 Y1 Z1` | echo extent | grid corners |
| `--dtm` | none | ESRI ASCII terrain grid; gives ground echoes and `distance_from_ground` |
| `--ground-class` | none | class code of ground echoes |
| `--ground-distance` | 0.2 | echoes this close above the DTM are ground (m) |
| `--leaf-classes`, `--wood-classes` | none | class codes for leaf and wood echoes |
| `--class-attr` | classification | echo attribute holding class codes |
| `--weighting` | equal | share of a pulse per echo: equal, full, first, relative, strongest |
| `--attenuation` | fpl | one or more of fpl, ppl, transmittance, bailey |
| `--laser` / `--beam D DIV` | none | scanner name, or beam diameter (m) and divergence (rad) |
| `--lad`, `--lad-params` | spherical | analytic leaf angle distribution |
| `--inclination`, `--iad CSV` | off | per-tree inclination distributions, optionally written to CSV |
| `--occlusion` | off | trace beyond each pulse's last echo |
| `--flat-top` | off | start paths in each column's top voxel at the highest echo |
| `--neighbour-priors N` | 0 | top up voxels with fewer weighted beams from their neighbours |
| `--subvoxel-split N` | 0 | N³ sub-voxel grid for `exploration_rate` |
| `--average-leaf-area` | 0.005 | mean leaf area (m²) of the free-path correction |
| `--qsm CSV ...` | none | QSM cylinder CSVs to rasterise as wood volume |
| `--write-empty` / `--filled-only` | off | also write unobserved voxels / only filled ones |

See [Ray-traced voxels](voxels.md) for what the options do.
