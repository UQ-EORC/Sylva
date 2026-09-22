"""Command-line interface: ``sylva <command> ...``.

Every command reads its input, writes its outputs and prints a one-line
summary. Errors print ``sylva: error: ...`` to stderr and exit with status 1,
so the commands can be chained in shell scripts and workflow managers.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import __version__, canopy, filters, ground, io, qsm, trees, voxels
from .raster import Raster
from .shots import Shots


def _cmd_info(args):
    cloud = io.read(args.input)
    lo, hi = cloud.bounds
    print(f"{args.input}: {len(cloud):,} points")
    print(f"  min: {lo}\n  max: {hi}")
    for k, v in cloud.attrs.items():
        print(f"  {k}: {v.dtype} [{v.min()} .. {v.max()}]")


def _cmd_convert(args):
    cloud = io.read(args.input)
    if args.voxel:
        cloud = filters.voxel_downsample(cloud, args.voxel)
    io.write(cloud, args.output)
    print(f"wrote {len(cloud):,} points to {args.output}")


def _cmd_ground(args):
    cloud = io.read(args.input)
    if args.method == "csf":
        cloud = ground.classify_ground_csf(cloud, cloth_resolution=args.resolution)
    else:
        cloud = ground.classify_ground_pmf(cloud, cell_size=args.resolution)
    dtm = ground.make_dtm(cloud, resolution=args.resolution)
    cloud = ground.normalize_height(cloud, dtm)
    io.write(cloud, args.output)
    if args.dtm:
        if args.dtm.lower().endswith((".tif", ".tiff")):
            dtm.to_geotiff(args.dtm)
        else:
            dtm.to_ascii_grid(args.dtm)
    print(f"{int(ground.ground_mask(cloud).sum()):,} ground points; wrote {args.output}")


def _cmd_trees(args):
    cloud = io.read(args.input)
    found = trees.detect_stems(cloud, min_radius=args.min_dbh / 2)
    labels = trees.segment_trees(cloud, found)
    trees.tree_heights(cloud, labels, found)
    found, labels = trees.prune_trees(found, labels, min_height=args.min_height)
    crowns = trees.crown_metrics_all(cloud, labels)
    if args.segment:
        io.write(cloud.with_attrs(tree_id=labels.astype(np.int32)), args.segment)
    rows = [{**t.as_dict(), **crowns.get(t.tree_id, {})} for t in found]
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    if rows:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if args.output:
        out.close()
        print(f"{len(found)} trees -> {args.output}")


def _cmd_chm(args):
    cloud = io.read(args.input)
    chm = ground.make_chm(cloud, resolution=args.resolution)
    if args.output.lower().endswith((".tif", ".tiff")):
        chm.to_geotiff(args.output)
    else:
        chm.to_ascii_grid(args.output)
    print(f"CHM {chm.shape} -> {args.output}; cover(>2m)={canopy.canopy_cover(chm.data):.2f}")


def _cmd_pad(args):
    cloud = io.read(args.input)
    z, pad = canopy.pad_profile_voxel(cloud, voxel_size=args.voxel)
    print("height,pad")
    for zz, p in zip(z, pad, strict=True):
        print(f"{zz:.2f},{p:.4f}")
    print(f"# PAI = {np.sum(pad) * args.voxel:.3f}", file=sys.stderr)


def _cmd_qsm(args):
    cloud = io.read(args.input)
    model = qsm.build_qsm(cloud, bin_length=args.bin_length)
    model.to_csv(args.output)
    for k, v in model.summary().items():
        print(f"{k}: {v}")


def _cmd_shots(args):
    shots = Shots.from_ray_cloud(io.read(args.input))
    shots.save(args.output, double=args.double)
    print(f"{shots.n_shots} pulses, {shots.n_echoes} echoes -> {args.output}")


def _cmd_voxel(args):
    # Shots files are streamed; ray clouds have to be loaded whole.
    streamed = args.input.lower().endswith(".parquet")
    shots = args.input if streamed else Shots.from_ray_cloud(io.read(args.input))
    bounds = None
    if args.bounds:
        bounds = (args.bounds[:3], args.bounds[3:])
    grid = voxels.ray_voxelize(
        shots, args.voxel, bounds,
        dtm=Raster.from_ascii_grid(args.dtm) if args.dtm else None,
        ground_class=args.ground_class, ground_distance=args.ground_distance,
        leaf_classes=args.leaf_classes, wood_classes=args.wood_classes, class_attr=args.class_attr,
        weighting=args.weighting, attenuation=args.attenuation, laser=args.laser, beam=args.beam,
        lad=args.lad, lad_params=args.lad_params, inclination=args.inclination,
        occlusion=args.occlusion, flat_top=args.flat_top,
        neighbour_prior_min_rays=args.neighbour_priors, subvoxel_split=args.subvoxel_split,
        average_leaf_area=args.average_leaf_area,
    )
    if args.qsm:
        grid.add_wood_volume(qsm.QSM.from_csv(f) for f in args.qsm)
    n = grid.write(args.output, include_unobserved=args.write_empty, filled_only=args.filled_only)
    if args.iad:
        grid.write_iad_csv(args.iad)
    n_shots = Shots.file_info(shots)["n_shots"] if streamed else shots.n_shots
    print(f"{n_shots} pulses -> {grid!r}; wrote {n} voxels to {args.output}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sylva",
                                description="Terrestrial laser scanning for forest ecology.",
                                epilog="Run `sylva <command> --help` for the options of a command.")
    p.add_argument("--version", action="version", version=f"sylva {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    fmt = {"formatter_class": argparse.ArgumentDefaultsHelpFormatter}

    s = sub.add_parser("info", help="print point count, bounds and attributes", **fmt)
    s.add_argument("input", help="point cloud (.las .laz .ply .xyz .txt .csv .pts .rxp)")
    s.set_defaults(func=_cmd_info)

    s = sub.add_parser("convert", help="convert between formats, optionally voxel-thinning", **fmt)
    s.add_argument("input", help="point cloud")
    s.add_argument("output", help="point cloud; format from the extension")
    s.add_argument("--voxel", type=float, default=None,
                   help="keep one point per voxel of this size (m)")
    s.set_defaults(func=_cmd_convert)

    s = sub.add_parser("ground", help="classify ground, build DTM and add height attribute", **fmt)
    s.add_argument("input", help="point cloud, z up")
    s.add_argument("output", help="cloud with classification (2 = ground) and height attributes")
    s.add_argument("--method", choices=["csf", "pmf"], default="csf",
                   help="cloth simulation or progressive morphological filter")
    s.add_argument("--resolution", type=float, default=0.5,
                   help="cloth / filter cell and DTM resolution (m)")
    s.add_argument("--dtm", help="also write the DTM (.tif needs rasterio, else .asc)")
    s.set_defaults(func=_cmd_ground)

    s = sub.add_parser("trees", help="detect stems and DBH from a height-normalised cloud", **fmt)
    s.add_argument("input", help="cloud with a height attribute (from `sylva ground`)")
    s.add_argument("-o", "--output",
                   help="CSV of trees, one row each with crown metrics (default: stdout)")
    s.add_argument("--min-dbh", type=float, default=0.05,
                   help="smallest stem diameter detected (m)")
    s.add_argument("--min-height", type=float, default=3.0,
                   help="drop candidates whose segment is lower than this (m)")
    s.add_argument("--segment", help="write a cloud with tree_id attribute to this path")
    s.set_defaults(func=_cmd_trees)

    s = sub.add_parser("chm", help="canopy height model from a height-normalised cloud", **fmt)
    s.add_argument("input", help="cloud with a height attribute")
    s.add_argument("output", help=".tif (needs rasterio) or .asc")
    s.add_argument("--resolution", type=float, default=0.5, help="cell size (m)")
    s.set_defaults(func=_cmd_chm)

    s = sub.add_parser("pad", help="plant area density profile (voxel method), CSV on stdout",
                       **fmt)
    s.add_argument("input", help="cloud with a height attribute")
    s.add_argument("--voxel", type=float, default=0.5, help="voxel size and layer thickness (m)")
    s.set_defaults(func=_cmd_pad)

    s = sub.add_parser("qsm", help="build a cylinder model of a single tree", **fmt)
    s.add_argument("input", help="one tree's wood points")
    s.add_argument("output", help="CSV of cylinders")
    s.add_argument("--bin-length", type=float, default=0.3, help="geodesic shell width (m)")
    s.set_defaults(func=_cmd_qsm)

    s = sub.add_parser("shots", help="convert a ray cloud to a sylva shots file (.parquet)", **fmt)
    s.add_argument("input", help="ray cloud with sx,sy,sz or nx,ny,nz attributes")
    s.add_argument("output", help="shots file (.parquet)")
    s.add_argument("--double", action="store_true", help="double-precision angles and ranges")
    s.set_defaults(func=_cmd_shots)

    s = sub.add_parser("voxel", help="ray-traced voxel grid (AMAPVox-style) from pulse data", **fmt)
    s.add_argument("input", help="shots file (.parquet, streamed) or ray cloud")
    s.add_argument("output", help=".vox (AMAPVox) or .txt")
    s.add_argument("--voxel", type=float, default=0.1, help="voxel size (m)")
    s.add_argument("--bounds", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
                   help="grid corners (default: extent of the echoes)")
    s.add_argument("--dtm", help="ESRI ASCII grid of terrain heights")
    s.add_argument("--ground-class", type=int, help="class code of ground echoes")
    s.add_argument("--ground-distance", type=float, default=0.2,
                   help="echoes this close above the DTM are ground (m)")
    s.add_argument("--leaf-classes", type=int, nargs="*", default=[],
                   help="class codes of leaf echoes")
    s.add_argument("--wood-classes", type=int, nargs="*", default=[],
                   help="class codes of wood echoes")
    s.add_argument("--class-attr", default="classification",
                   help="echo attribute holding class codes")
    s.add_argument("--weighting", default="equal",
                   choices=["equal", "full", "first", "relative", "strongest"],
                   help="share of a pulse carried by each echo")
    s.add_argument("--attenuation", nargs="+", default=["fpl"],
                   choices=["fpl", "ppl", "transmittance", "bailey"], help="attenuation estimators")
    s.add_argument("--laser", help="scanner name, e.g. VZ-400")
    s.add_argument("--beam", type=float, nargs=2, metavar=("DIAMETER", "DIVERGENCE"),
                   help="beam exit diameter (m) and divergence (rad), instead of --laser")
    s.add_argument("--lad", default="spherical", help="analytic leaf angle distribution")
    s.add_argument("--lad-params", type=float, nargs="*", default=[],
                   help="parameters of an ellipsoidal or beta distribution")
    s.add_argument("--inclination", action="store_true",
                   help="estimate per-tree inclination angle distributions")
    s.add_argument("--iad", help="write the per-tree inclination distributions to this CSV")
    s.add_argument("--occlusion", action="store_true", help="trace beyond each pulse's last echo")
    s.add_argument("--flat-top", action="store_true",
                   help="start paths in each column's top voxel at the highest echo")
    s.add_argument("--neighbour-priors", type=int, default=0, metavar="MIN_RAYS",
                   help="top up voxels with fewer weighted beams from their neighbours (0 = off)")
    s.add_argument("--subvoxel-split", type=int, default=0,
                   help="N for an N^3 sub-voxel exploration grid (0 = off)")
    s.add_argument("--average-leaf-area", type=float, default=0.005,
                   help="mean leaf area (m2) of the free path correction (0 = off)")
    s.add_argument("--qsm", nargs="*", help="QSM cylinder CSVs to rasterise as wood volume")
    s.add_argument("--write-empty", action="store_true", help="also write unobserved voxels")
    s.add_argument("--filled-only", action="store_true", help="only write voxels holding echoes")
    s.set_defaults(func=_cmd_voxel)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, ImportError) as e:
        p.exit(1, f"sylva: error: {e}\n")


if __name__ == "__main__":
    main()
