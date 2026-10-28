#!/bin/bash
# Sylva plot pipeline: DTM and heights, tree segmentation, QSMs with wood
# meshes, leaf meshes, and understorey voxels, from one merged plot cloud.
#
#   ./plot_pipeline.sh PLOT.laz OUT_DIR [SHOTS.parquet]
#
# Every stage writes into OUT_DIR under the input's name and is skipped when
# its output already exists, so a run can be resumed. Settings are taken from
# the environment, e.g. `GRAPH_VOXEL=0.03 MIN_POINTS=5000 ./plot_pipeline.sh ...`:
#
#   THIN=0.05           thinning (m) of the copy the ground filter and PAD run on
#   DTM_RES=0.5         ground filter cell and DTM resolution (m)
#   GRAPH_VOXEL=0.05    thinning (m) of the segmentation graph (0.03 default of
#                       sylva trees; 0.05 is what a plot of a few hundred
#                       million points needs to fit in memory)
#   MIN_DBH=0.05        smallest stem detected (m)
#   MIN_HEIGHT=3        trees lower than this (m) are dropped after segmentation
#   MIN_SLENDERNESS=10  stems wider than 0.2 m less than this many times as tall as
#                       wide are dropped (understorey clumps fitted as stems); 0 keeps them
#   MIN_QUALITY_SHORT=0.3  stems found in fewer than SHORT_SLICES layers with a circle fit
#   SHORT_SLICES=5      of lower quality than this are dropped; 0 keeps them
#   MIN_POINTS=2000     trees with fewer points get no QSM or leaves
#   STEM_CAP=1.5        main-stem circles wider than this times the detected DBH
#                       are set aside (foliage sheathing the bole); 0 turns it off
#   LEAF_LENGTH=0.08    leaf blade size (m) for the leaf meshes
#   LEAF_WIDTH=0.04
#   UNDERSTOREY_TOP=3   height (m) below which the PAD profile is the understorey
#   VOXEL=0.25          ray-traced voxel size (m), only with a shots file
#
# The ray-traced voxels need the pulses: a sylva shots file (.parquet) in the
# plot's frame, as the third argument. Without one the understorey comes from
# the point-based PAD profile alone.
set -euo pipefail

PLOT=${1:?usage: plot_pipeline.sh PLOT.laz OUT_DIR [SHOTS.parquet]}
OUT=${2:?usage: plot_pipeline.sh PLOT.laz OUT_DIR [SHOTS.parquet]}
SHOTS=${3:-}

THIN=${THIN:-0.05}
DTM_RES=${DTM_RES:-0.5}
GRAPH_VOXEL=${GRAPH_VOXEL:-0.05}
MIN_DBH=${MIN_DBH:-0.05}
MIN_HEIGHT=${MIN_HEIGHT:-3}
MIN_SLENDERNESS=${MIN_SLENDERNESS:-10}
MIN_QUALITY_SHORT=${MIN_QUALITY_SHORT:-0.3}
SHORT_SLICES=${SHORT_SLICES:-5}
MIN_POINTS=${MIN_POINTS:-2000}
STEM_CAP=${STEM_CAP:-1.5}
LEAF_LENGTH=${LEAF_LENGTH:-0.08}
LEAF_WIDTH=${LEAF_WIDTH:-0.04}
UNDERSTOREY_TOP=${UNDERSTOREY_TOP:-3}
VOXEL=${VOXEL:-0.25}

# The Python CLI has every command; the Rust one segments in a third of the
# memory, so it takes the segmentation when it is installed.
SYLVA=${SYLVA:-$HOME/Code/sylva/.venv/bin/sylva}
PY=${PY:-$HOME/Code/sylva/.venv/bin/python}
SYLVA_RS=${SYLVA_RS:-$HOME/.cargo/bin/sylva}
[ -x "$SYLVA_RS" ] || SYLVA_RS=$SYLVA

NAME=$(basename "${PLOT%.*}")
mkdir -p "$OUT"
LOG=$OUT/$NAME.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

THINNED=$OUT/${NAME}_thin.laz
NORM_THIN=$OUT/${NAME}_thin_norm.laz
DTM=$OUT/${NAME}_dtm.asc
NORM=$OUT/${NAME}_norm.laz
TREES=$OUT/${NAME}_trees.csv
SEG=$OUT/${NAME}_segmented.laz
QSM_CSV=$OUT/${NAME}_qsm.csv
CYL=$OUT/${NAME}_cylinders
WOOD_MESH=$OUT/${NAME}_wood_meshes
LEAF_MESH=$OUT/${NAME}_leaf_meshes
PAD=$OUT/${NAME}_pad.csv
UNDER=$OUT/${NAME}_understorey_pad.csv
VOX=$OUT/${NAME}_voxels_${VOXEL}m.vox

# 1. DTM from a thinned copy, then height above ground on every point.
if [ ! -s "$NORM" ]; then
    say "1. ground: thinning to $THIN m"
    [ -s "$THINNED" ] || "$SYLVA" convert "$PLOT" "$THINNED" --voxel "$THIN"
    say "1. ground: PMF at $DTM_RES m, DTM -> $DTM"
    [ -s "$DTM" ] || "$SYLVA" ground "$THINNED" "$NORM_THIN" --method pmf --resolution "$DTM_RES" --dtm "$DTM"
    say "1. ground: heights on the full cloud -> $NORM"
    "$PY" - "$PLOT" "$DTM" "$NORM" <<'EOF'
import sys
import numpy as np
from sylva import Raster, ground, io

src, dtm_path, out = sys.argv[1:4]
dtm = Raster.from_ascii_grid(dtm_path)
cloud = io.read(src, attrs=[])                       # coordinates only
h = ground.normalize_height(cloud, dtm).attrs["height"]
print(f"   {len(cloud):,} points; height p1 {np.percentile(h, 1):.2f} m, "
      f"p99.9 {np.percentile(h, 99.9):.1f} m, below -0.3 m {100 * (h < -0.3).mean():.2f} %")
io.write_annotated(src, out, {"height": h.astype(np.float32)}, origin=cloud.bounds[0])
EOF
fi

# 2. Stems and tree segmentation.
if [ ! -s "$SEG" ]; then
    say "2. trees: segmentation on a $GRAPH_VOXEL m graph -> $SEG"
    "$SYLVA_RS" trees "$NORM" -o "$TREES" --min-dbh "$MIN_DBH" --min-height "$MIN_HEIGHT" --min-slenderness "$MIN_SLENDERNESS" --min-quality-short "$MIN_QUALITY_SHORT" --short-slices "$SHORT_SLICES" --segment "$SEG" --voxel "$GRAPH_VOXEL" | tee -a "$LOG"
fi

# 3. A QSM per tree, with cylinder tables and wood meshes.
if [ ! -s "$QSM_CSV" ]; then
    say "3. qsm: a QSM per tree with at least $MIN_POINTS points -> $QSM_CSV"
    "$SYLVA" qsm-plot "$SEG" "$QSM_CSV" --stems "$TREES" --stem-radius-cap "$STEM_CAP" --min-points "$MIN_POINTS" --cylinders "$CYL" --meshes "$WOOD_MESH" | tee -a "$LOG"
fi

# 4. Leaf meshes: wood and leaves of every modelled tree as one OBJ.
say "4. leaves: $LEAF_LENGTH x $LEAF_WIDTH m blades on the QSMs -> $LEAF_MESH"
mkdir -p "$LEAF_MESH"
"$PY" - "$SEG" "$CYL" "$LEAF_MESH" "$MIN_POINTS" "$LEAF_LENGTH" "$LEAF_WIDTH" <<'EOF' | tee -a "$LOG"
import glob, os, sys
import numpy as np
from sylva import io, leaves
from sylva.qsm import QSM

seg, cyl_dir, out_dir, min_points, length, width = sys.argv[1:7]
min_points, length, width = int(min_points), float(length), float(width)
cloud = io.read(seg, attrs=["tree_id"])
ids = cloud.attrs["tree_id"].astype(np.int64)
order = np.argsort(ids, kind="stable")
sorted_ids = ids[order]
done = skipped = failed = 0
for path in sorted(glob.glob(os.path.join(cyl_dir, "tree*.csv"))):
    tid = int(os.path.basename(path)[4:-4])
    out = os.path.join(out_dir, f"tree{tid}.obj")
    if os.path.exists(out):
        done += 1
        continue
    lo, hi = np.searchsorted(sorted_ids, [tid, tid + 1])
    idx = order[lo:hi]
    if len(idx) < min_points:
        skipped += 1
        continue
    try:
        tree = cloud[idx]
        wood = leaves.classify_leaf_wood(tree)
        foliage = tree[~wood]
        if len(foliage) < 100:
            skipped += 1
            continue
        model = QSM.from_csv(path)
        angles = leaves.leaf_angle_distribution(foliage)
        area = leaves.leaf_area_density(foliage, voxel_size=0.25)
        mesh = leaves.add_leaves(model, area, angles, leaf_points=foliage,
                                 leaf_length=length, leaf_width=width)
        leaves.write_tree_obj(out, model, mesh)
        done += 1
    except Exception as exc:                        # one tree must not stop the plot
        failed += 1
        print(f"   tree {tid}: {exc}")
print(f"   {done} trees with leaves, {skipped} skipped, {failed} failed -> {out_dir}")
EOF

# 5. Understorey: the PAD profile by height, and ray-traced voxels when the
#    pulses are at hand.
if [ ! -s "$PAD" ]; then
    say "5. understorey: PAD profile of the $THIN m cloud -> $PAD"
    "$SYLVA" pad "$NORM_THIN" --voxel 0.5 > "$PAD"
fi
awk -F, -v top="$UNDERSTOREY_TOP" 'NR == 1 || ($1 + 0 < top && $1 !~ /^#/)' "$PAD" > "$UNDER"
say "5. understorey: layers below $UNDERSTOREY_TOP m -> $UNDER"
if [ -n "$SHOTS" ]; then
    if [ ! -s "$VOX" ]; then
        say "5. understorey: ray-traced $VOXEL m voxels from $SHOTS -> $VOX"
        "$SYLVA" voxel "$SHOTS" "$VOX" --voxel "$VOXEL" --dtm "$DTM" --occlusion --attenuation fpl | tee -a "$LOG"
    fi
else
    say "5. understorey: no shots file given, so no ray-traced voxels (pass a plot-frame .parquet as the third argument)"
fi

say "done: $TREES, $SEG, $QSM_CSV, $CYL, $WOOD_MESH, $LEAF_MESH, $UNDER"
