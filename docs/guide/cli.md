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
sylva qsm-plot plot_trees.laz trees.csv --cylinders qsms/ --meshes meshes/
sylva coreg survey.PROJ -o survey_coreg/ --merged survey.laz
```

## Where the outputs go

Every output is optional: leave it out and the command writes beside its
input, under the input's own name. A whole plot, without naming a single
output path:

```bash
sylva ground plot.laz                     # -> plot_norm.laz
sylva trees plot_norm.laz --segment       # -> plot_norm_trees.csv, plot_norm_segmented.laz
sylva qsm-plot plot_norm_segmented.laz --cylinders --meshes
                                          # -> ..._trees.csv, ..._cylinders/, ..._meshes/
```

| Command | Default output |
|---|---|
| `ground` | `<input>_norm<ext>`, plus `--dtm PATH` if asked |
| `trees` | `<input>_trees.csv`; `--segment` alone gives `<input>_segmented<ext>`; `-o -` writes stdout |
| `chm` | `<input>_chm.asc` |
| `qsm` | `<input>_qsm.csv` |
| `qsm-plot` | `<input>_trees.csv`; `--cylinders` / `--meshes` alone give `<input>_cylinders/` and `<input>_meshes/` |
| `shots` | `<input>.parquet` |
| `voxel` | `<input>.vox` |
| `coreg` | `<project>_coreg/` (`transforms.json`, `report.txt`, one `.dat` per scan) |

`convert` is the exception: its format comes from the output extension, so it
needs one. `pad` prints to stdout by design.

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
| `--meshes DIR` | none | also write a surface mesh per tree, fused with its buttress |
| `--mesh-format ply\|obj` | `ply` | format for `--meshes` |
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

### `coreg`

`sylva coreg INPUT... [-o DIR] [options]`: registers the scan positions of a
survey from the trees, with no targets or initial alignment
(:mod:`sylva.coreg`, a port of tlsalign). `INPUT` is a RiSCAN PRO project or
a scanner `.PROJ` directory, or a list of scan files. For a project, tilted
scans are levelled with the scanner's attitude (or, failing that, the rotation
of the SOP), the scanner's GNSS fixes skip pairs too far apart, and its
reflective targets are used ahead of stems where scans share three.

Writes `transforms.json` (a `world_from_scan` matrix per scan, with the
quality of every pair), `report.txt` (every scan and pair, and the stem
agreement per pair under the final poses: the median distance between the
same tree seen from both scans, a check that needs no ground truth) and a
`<scan>.dat` matrix per registered scan.

| Option | Default | What |
|---|---|---|
| `--reference NAME` | first scan | scan whose frame is the world frame |
| `--level` | `auto` | `attitude`, `sop` or `none`; `auto` is the attitude, else the SOP rotation |
| `--sop-priors` | off | use the SOPs as priors: refuse results that move a scanner more than 5 m from its SOP, and place scans with too few stems from their SOP |
| `--refine` | off | joint multi-view refinement of all poses |
| `--no-reflectors` | off | ignore reflective targets |
| `--max-pair-distance` | 40 | skip pairs further apart by GNSS (m) |
| `--min-range`, `--max-range` | none | echo range bounds (m) |
| `--min-deviation`, `--max-deviation` | none | pulse deviation bounds |
| `--min-reflectance`, `--max-reflectance` | none | reflectance bounds (dB) |
| `--min-amplitude`, `--max-amplitude` | none | amplitude bounds (dB) |
| `--riscan-export-settings FILE` | none | bounds from a RiSCAN PRO export filter settings file (`attribute, min, max` per line); explicit bounds override it |
| `--riscan-filter` | `none` | RiSCAN PRO's RXP import filter: `current` drops echoes within 0.5 m of the scanner (a current import, 99.7 % agreement); `legacy` also drops the weak, isolated echoes the older conversion discarded (a fifth to a third of a scan) |
| `--trust-reflectors N` | 5 | accept a reflector match of at least N targets within 3 cm even when ICP fails its fitness test (scans far apart share targets before they share surface); 0 always asks ICP to agree, as tlsalign |
| `--workers` | 0 | scans and pairs processed at once; 0 picks from cores and memory |
| `--merged PATH` | none | also write the merged, registered cloud |
| `--voxel` | 0.02 | thinning of the merged cloud (m) |
