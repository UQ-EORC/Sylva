# Command line

Installing the Python package provides a `sylva` command. Each command reads
its input, writes its outputs and prints a one-line summary. On error it
prints `sylva: error: ...` to stderr and exits with status 1, so commands can
be chained in shell scripts and workflow managers. `sylva <command> --help`
shows every option with its default, and `sylva --version` prints the
version.

```bash
sylva info plot.laz
sylva reproject plot.laz plot_lonlat.laz --to EPSG:7844
sylva transform ScanPos001.laz ScanPos001_project.laz --matrix ScanPos001.DAT
sylva ground plot.laz plot_norm.laz --dtm dtm.tif
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva chm plot_norm.laz chm.tif
sylva pad plot_norm.laz --voxel 0.5 > pad.csv
sylva shots rays.laz plot.parquet                  # ray cloud -> shots file
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2 --laser VZ-400 --attenuation fpl ppl
sylva qsm tree.ply tree_qsm.csv
sylva qsm-plot plot_trees.laz trees.csv --cylinders qsms/ --meshes meshes/
sylva coreg survey.PROJ -o survey_coreg/ --merged survey.laz
sylva als-catalog tiles/                           # airborne tiles: headers only
sylva als-ground tiles/ ground/ && sylva als-dtm ground/ dtm.tif --resolution 1
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

`convert`, `reproject` and `transform` are the exceptions: the output format
comes from its extension, so they need one. So are the `als-*` commands,
which read a directory of tiles and write a directory or a raster. `pad`
prints to stdout by design.

`crates/sylva-cli` builds a standalone Rust binary with the same commands
and no Python dependency (`cargo install --path crates/sylva-cli`). Its
options can differ in detail, so check its `--help`.

## Commands

### `info`

`sylva info INPUT`: prints the point count, bounds, the CRS (LAS/LAZ
headers) and each attribute's type and range.

### `convert`

`sylva convert INPUT OUTPUT [--voxel SIZE]`: converts between any readable
and writable formats (by extension). `--voxel` keeps one point per voxel of
that size (m).

### `reproject`

`sylva reproject INPUT OUTPUT --to CRS [--from CRS]`: transforms the points
into another coordinate reference system with
[`sylva.coords.reproject`](coordinates.md#reprojecting). A CRS is an EPSG code
(`EPSG:7855`, or a compound `EPSG:7855+5711`), a PROJ string or WKT (quote
both on the command line). `--from` defaults to the CRS in the input's
LAS/LAZ header; other formats carry none, so need it. A LAS/LAZ output stores
the new CRS. The summary names the kind of transformation, and an
approximate one (a datum change without known parameters, a vertical datum
change) also prints a warning saying what was not applied.

```bash
sylva reproject plot.laz plot_wgs84.laz --to EPSG:4326          # CRS from the header
sylva reproject plot.ply plot_mga55.laz --from EPSG:28355 --to EPSG:7855
```

### `transform`

`sylva transform INPUT OUTPUT (--matrix FILE | --translate DX DY DZ | --rotate DEG [--axis x|y|z] [--about X Y Z])`:
applies one rigid transformation and keeps every attribute and the CRS.

- `--matrix FILE` applies a 4x4 matrix stored as 16 numbers in row-major
  order: a RiSCAN SOP or POP `.DAT`, or a `.dat` written by `sylva coreg`.
- `--translate DX DY DZ` shifts every point, e.g. to move projected
  coordinates to a local origin and back.
- `--rotate DEG` rotates by `DEG` degrees about `--axis` (default `z`),
  counter-clockwise seen from the positive axis, through `--about`
  (default the origin).

```bash
sylva transform ScanPos003.rxp ScanPos003.laz --matrix plot_coreg/ScanPos003.dat
sylva transform plot.laz plot_local.laz --translate -512000 -5412000 0
sylva transform plot.laz plot_turned.laz --rotate 12.5 --about 512025 5412025 0
```

To put a whole survey in one frame with a `transforms.json` or a folder of
`.DAT` files, use [`sylva.coords.apply_transforms`](coordinates.md#applying-registration-results).

### `ground`

`sylva ground INPUT OUTPUT [--method csf|pmf] [--resolution 0.5] [--dtm PATH]`

Classifies ground (`classification` 2), builds a DTM at `--resolution` (m),
and writes the cloud with a `height` attribute. `--dtm` also writes the
terrain (`.tif`, otherwise `.asc`).

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
(:mod:`sylva.coreg`). `INPUT` is a RiSCAN PRO project or
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
| `--trust-reflectors N` | 5 | accept a reflector match of at least N targets within 3 cm even when ICP fails its fitness test (scans far apart share targets before they share surface); 0 always asks ICP to agree |
| `--workers` | 0 | scans and pairs processed at once; 0 picks from cores and memory |
| `--merged PATH` | none | also write the merged, registered cloud |
| `--voxel` | 0.02 | thinning of the merged cloud (m) |

### Airborne tiles: `als-catalog`, `als-ground`, `als-dtm`, `als-chm`, `als-normalize`

These work on a directory of LAS/LAZ tiles (see [Airborne lidar
tiles](als.md)), processed in buffered chunks so that tile edges do not show.

```bash
sylva als-catalog tiles/ [--pattern '*.la[sz]'] [--recursive] [--tolerance 1] [--strict]
sylva als-ground tiles/ ground/ [--method csf|pmf] [--resolution 0.5] [--last-returns]
sylva als-dtm ground/ dtm.tif [--resolution 1] [--method lowest|tin|natural|idw]
sylva als-chm ground/ chm.tif [--resolution 0.5] [--dtm-resolution 1] [--dtm DTM.asc] [--normalized]
sylva als-normalize ground/ normalised/ [--dtm-resolution 1] [--dtm DTM.asc] [--replace-z]
```

`als-catalog` prints the catalogue report: extent, points, density, formats,
CRS and every problem found (missing or unreadable files, mixed CRS or point
formats, overlapping tiles, holes); with `--strict` it exits with status 1 if
there is any. `als-ground` writes the tiles with ground classified;
`als-dtm` and `als-chm` write one raster for the whole area (`.tif`, or
`.asc` with a `.prj`); `als-chm` normalises on the fly from the
ground points unless given `--dtm` or `--normalized` (tiles whose z is
already height); `als-normalize` writes the tiles with a `height` attribute,
or with z replaced by it (`--replace-z`, the elevation kept as
`elevation`).

| Option | Default | What |
|---|---|---|
| `--pattern` | `*.la[sz]` | file name pattern within the directory (case-insensitive) |
| `--chunk-size M` | one chunk per tile | process squares of this size instead |
| `--buffer M` | 20 | band of neighbouring points read around each chunk |
| `--workers N` | one per CPU | chunks at once; fewer if memory is short |

`als-metrics` and `als-plot-metrics` compute area-based metrics as one
raster per metric or a CSV table of plots; see [Area-based ALS
metrics](als_metrics.md#command-line).

### Plots in tiles: `tiles-from-scans`, `tiles-thin`, `tiles-filter`, `tiles-features`, `tiles-stems`

These build and process a plot kept as square tiles, one tile at a time,
with results identical to those on the whole cloud (see [Plots in
tiles](tiles.md)). Ground, DTM and heights use `als-ground`, `als-dtm` and
`als-normalize` above on the same tiles.

```bash
sylva tiles-from-scans scans/*.laz tiles/ [--tile-size 10] [--voxel 0.02] [--transforms sop.npy]
                       [--bounds XMIN YMIN XMAX YMAX] [--format laz] [--scale 0.001] [--epsg CODE]
sylva tiles-thin tiles/ thinned/ --voxel 0.05
sylva tiles-filter tiles/ clean/ [--method sor|ror] [--k 8] [--std-ratio 2] [--radius 0.1]
                   [--min-neighbors 4] [--classify]
sylva tiles-features tiles/ features/ [--feature normals|shape] [--k 12]
sylva tiles-stems normalised/ stems.csv [--height-attribute height] [--min-arc 0]
```

`tiles-from-scans` takes the scans in order (with `--transforms`, a `.npy`
of one 4x4 matrix per scan) and writes tiles thinned on one global voxel
grid, with a `scan_id` attribute. `tiles-filter` removes noise (or
classifies it 7 with `--classify`); `--method sor` is CloudCompare's
statistical filter with its global threshold. `tiles-features` adds
`normal_x`, `normal_y`, `normal_z` or `planarity` and `linearity` to every
point; `tiles-stems` writes the stems as a CSV table. Each prints the most
points it held at once. They take `--pattern`, `--buffer` (1 m, or 2 m for
`tiles-stems`; `tiles-thin` needs none) and `--workers`.

### TLS and ALS together: `fusion-register`, `fusion-trees`, `fusion-upscale`

```bash
sylva fusion-register plot.laz tiles/ reg.json [--initial gnss.txt] [--search-radius 10]
                      [--heading-range 180] [--refine all|ground|none] [--transformed plot_map.laz]
sylva fusion-trees tls_trees.csv als_trees.csv linked.csv [--crowns crowns.geojson]
                   [--registration reg.json] [--als-output als_linked.csv]
sylva fusion-upscale plots.csv metrics/ out/ --response agb --predictors zq95,cover [--model loglog]
```

`fusion-register` places a TLS plot (ground classified) on a directory of
ALS tiles or one ALS file with [`fusion.register`](../api/fusion.md),
writes the transform, residuals and uncertainty as JSON and prints the
report; `--initial` is a text file with the 4 x 4 matrix of a rough
placement, and `--transformed` also writes the moved cloud.
`fusion-trees` links a CSV of TLS trees to the trees and crowns written by
`als-trees`, and writes each TLS tree's status, ALS tree and combined
height. `fusion-upscale` fits a plot value on ALS metrics and writes
`mean`, `se`, `lower`, `upper` and `extrapolated` rasters over the grid of
`als-metrics`. See [TLS and ALS together](fusion.md#command-line).

### Airborne trees: `als-trees`

```bash
sylva als-trees ground/ trees.csv [--crowns crowns.geojson] [--labelled labelled/]
                [--method dalponte2016|watershed|li2012|tops] [--resolution 0.5]
                [--window 5 | --window-linear INTERCEPT SLOPE MIN MAX] [--hmin 2]
                [--dtm DTM.asc | --normalized] [--buffer 30]
```

Finds the trees of a directory of tiles with
[`als.find_trees`](../api/als.md) (see [Airborne trees](als_trees.md)) and
writes one CSV row per tree (`id`, `x`, `y`, `height`, `crown_area`,
`n_points`); `--crowns` also writes the crown polygons as GeoJSON (the tree
tops with `--method tops`) and `--labelled` the tiles with each point's tree
id in a `tree_id` attribute. Heights come from the ground points unless
given `--dtm` or `--normalized`. The detection and segmentation settings
(`--shape`, `--tops-from`, `--smooth`, `--th-tree`, `--th-seed`, `--th-cr`,
`--max-cr`, `--dt1`, `--dt2`, `--R`, `--Zu`, `--speed-up`, `--hull`,
`--concavity`, `--min-point-height`) are those of the function; the
buffer is 30 m by default, and it takes `--pattern`, `--chunk-size` and
`--workers` as the other `als-*` commands do.

### Airborne change: `als-change`, `als-tree-change`

```bash
sylva als-change 2019/ 2024/ change/ [--surface chm|dsm|dtm] [--resolution 1]
                 [--align [--stable-classes 2,6] [--block-size 100] [--model field]]
                 [--harmonise [--density-cell 10]] [--first-returns] [--confidence 0.95]
                 [--gaps [--gap-height 2] [--gap-min-area 10]] [--years 5] [--format asc|tif]
sylva als-tree-change 2019/ 2024/ trees.csv [--resolution 0.5] [--method dalponte2016]
                 [--window 5 | --window-linear INTERCEPT SLOPE MIN MAX] [--hmin 2]
                 [--align] [--harmonise] [--max-distance 1.5] [--grid DIR --grid-resolution 50]
```

Compare two surveys of one area (see [Change between airborne
surveys](change_als.md)). `als-change` writes the two surfaces, their
difference, its level of detection and the class of each cell as rasters,
with `--align` the alignment's blocks (`alignment.csv`) and with `--gaps`
the gaps of both surveys with their status (`gaps_a.geojson`,
`gaps_b.geojson`). `als-tree-change` finds the trees of both surveys (on
thinned copies with `--harmonise`), compares them and writes one CSV row
per tree, with `--grid` also the per-cell totals as rasters. Both print
their reports and take `--pattern`, `--chunk-size`, `--buffer` and
`--workers`.
