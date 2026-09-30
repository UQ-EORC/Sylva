# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Command-line interface: ``sylva <command> ...``.

Every command reads its input, writes its outputs and prints a one-line
summary. Errors print ``sylva: error: ...`` to stderr and exit with status 1,
so the commands can be chained in shell scripts and workflow managers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from . import __version__, als, canopy, coreg, filters, ground, io, progress, qsm, riscan, trees, voxels
from .raster import Raster
from .shots import Shots


def _beside(path: str, suffix: str, ext: str | None = None) -> str:
    """A path beside ``path``: its stem plus ``suffix``, and ``ext`` if given.

    Every command writes next to its input unless told otherwise, so
    ``sylva ground plot.laz`` writes ``plot_norm.laz`` in the same folder.
    """
    p = Path(path)
    return str(p.with_name(p.stem + suffix + (ext if ext is not None else p.suffix)))


def _cmd_info(args):
    cloud = io.read(args.input)
    lo, hi = cloud.bounds
    print(f"{args.input}: {len(cloud):,} points")
    print(f"  min: {lo}\n  max: {hi}")
    if cloud.crs is not None:
        from .coords import crs_info

        try:
            info = crs_info(cloud.crs)
            print(f"  crs: {info.label} ({info.name})" if info.label != info.name else f"  crs: {info.name}")
        except ValueError:
            print(f"  crs: {cloud.crs[:80]}")
    for k, v in cloud.attrs.items():
        print(f"  {k}: {v.dtype} [{v.min()} .. {v.max()}]")


def _cmd_convert(args):
    cloud = io.read(args.input)
    if args.voxel:
        cloud = filters.voxel_downsample(cloud, args.voxel)
    io.write(cloud, args.output)
    print(f"wrote {len(cloud):,} points to {args.output}")


def _cmd_reproject(args):
    from . import coords

    cloud = io.read(args.input)
    src = args.src if args.src else cloud.crs
    if src is None:
        raise ValueError(f"{args.input} declares no CRS; give it with --from")
    out = coords.reproject(cloud, args.to, src_crs=src)
    io.write(out, args.output)
    t = coords.transformation(src, args.to)
    print(f"reprojected {len(out):,} points to {coords.crs_info(args.to).label} ({t.kind}); "
          f"wrote {args.output}")


def _cmd_transform(args):
    cloud = io.read(args.input)
    if args.matrix:
        cloud = cloud.transform(io.read_matrix_file(args.matrix))
        what = f"matrix {args.matrix}"
    elif args.translate:
        cloud = cloud.translate(*args.translate)
        what = "translation by ({:g}, {:g}, {:g})".format(*args.translate)
    else:
        cloud = cloud.rotate(args.rotate, axis=args.axis, about=args.about)
        about = "the origin" if args.about is None else "({:g}, {:g}, {:g})".format(*args.about)
        what = f"rotation by {args.rotate:g} degrees about {args.axis} through {about}"
    io.write(cloud, args.output)
    print(f"applied {what} to {len(cloud):,} points; wrote {args.output}")


def _cmd_ground(args):
    args.output = args.output or _beside(args.input, "_norm")
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
    args.output = args.output or _beside(args.input, "_trees", ".csv")
    if args.segment == "":                       # --segment with no path
        args.segment = _beside(args.input, "_segmented")
    cloud = io.read(args.input)
    found = trees.detect_stems(cloud, min_radius=args.min_dbh / 2)
    labels = trees.segment_trees(cloud, found)
    trees.tree_heights(cloud, labels, found)
    found, labels = trees.prune_trees(found, labels, min_height=args.min_height)
    crowns = trees.crown_metrics_all(cloud, labels)
    if args.segment:
        io.write(cloud.with_attrs(tree_id=labels.astype(np.int32)), args.segment)
    rows = [{**t.as_dict(), **crowns.get(t.tree_id, {})} for t in found]
    to_file = args.output != "-"
    out = open(args.output, "w", newline="") if to_file else sys.stdout
    if rows:
        w = csv.DictWriter(out, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if to_file:
        out.close()
        print(f"{len(found)} trees -> {args.output}")


def _cmd_chm(args):
    args.output = args.output or _beside(args.input, "_chm", ".asc")
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
    args.output = args.output or _beside(args.input, "_qsm", ".csv")
    cloud = io.read(args.input)
    model = qsm.build_qsm(cloud, bin_length=args.bin_length)
    model.to_csv(args.output)
    for k, v in model.summary().items():
        print(f"{k}: {v}")


def _cmd_qsm_plot(args):
    args.output = args.output or _beside(args.input, "_trees", ".csv")
    if args.cylinders == "":                     # --cylinders with no path
        args.cylinders = _beside(args.input, "_cylinders", "")
    if args.meshes == "":
        args.meshes = _beside(args.input, "_meshes", "")
    cloud = io.read(args.input)
    if args.tree_attr not in cloud.attrs:
        raise KeyError(f"{args.input} has no '{args.tree_attr}' attribute; "
                       f"segment it first (sylva trees --segment)")
    labels = cloud.attrs[args.tree_attr].astype(int)
    plot = qsm.build_plot(cloud, labels, voxel_size=args.voxel, wood=not args.no_wood,
                          buttress=args.buttress, min_points=args.min_points,
                          bin_length=args.bin_length)
    plot.to_csv(args.output)
    if args.cylinders:
        plot.write_cylinders(args.cylinders)
    if args.meshes:
        n = plot.write_meshes(args.meshes, fmt=args.mesh_format)
        print(f"{len(n)} meshes -> {args.meshes}")
    print(f"{len(plot)} QSMs, {len(plot.skipped)} skipped, "
          f"{plot.total_volume:.3f} m3 of wood -> {args.output}")
    for tid, why in list(plot.skipped.items())[:5]:
        print(f"  skipped {tid}: {why}")


def _cmd_shots(args):
    args.output = args.output or _beside(args.input, "", ".parquet")
    shots = Shots.from_ray_cloud(io.read(args.input))
    shots.save(args.output, double=args.double)
    print(f"{shots.n_shots} pulses, {shots.n_echoes} echoes -> {args.output}")


def _cmd_voxel(args):
    args.output = args.output or _beside(args.input, "", ".vox")
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


def _cmd_coreg(args):
    inputs = [Path(x) for x in args.inputs]
    positions = None
    if len(inputs) == 1 and inputs[0].is_dir():
        project = riscan.read_riscan_project(inputs[0])
        positions = project.with_scans(require_sop=False)
        if not positions:
            raise ValueError(f"no scans found in {inputs[0]}")
        clouds, names = [p.rxp for p in positions], [p.name for p in positions]
        default_out = inputs[0].with_name(inputs[0].name + "_coreg")
    else:
        clouds, names = inputs, [p.stem for p in inputs]
        default_out = inputs[0].with_name("coreg")
    out = Path(args.output) if args.output else default_out
    cfg = coreg.CoregConfig(refine_multiview=args.refine, use_reflectors=not args.no_reflectors,
                            max_pair_distance=args.max_pair_distance, workers=args.workers,
                            verbose=not args.quiet)
    cfg.riscan_filter = args.riscan_filter
    cfg.riegl_options = coreg.reading_options(
        args.riscan_export_settings,
        **{k: getattr(args, k) for k in ("min_range", "max_range", "min_deviation", "max_deviation",
                                         "min_reflectance", "max_reflectance", "min_amplitude",
                                         "max_amplitude")})
    if args.trust_reflectors is not None:
        cfg.trusted_reflector_matches = args.trust_reflectors
    if args.reference is not None:
        ref = args.reference
        cfg.reference_scan = names.index(ref) if ref in names else int(ref)
    levelling = reflectors = gnss = priors = None
    if positions is not None:
        def level(p):
            if args.level in ("auto", "attitude") and p.levelling is not None:
                return p.levelling
            if args.level in ("auto", "sop") and p.sop is not None:
                m = np.eye(4)
                m[:3, :3] = p.sop[:3, :3]
                return m
            return None
        levelling = [level(p) for p in positions]
        reflectors = [p.reflectors() for p in positions]
        gnss = riscan.gnss_to_local([p.gnss for p in positions])
        if not np.isfinite(gnss).any():
            gnss = None
        if args.sop_priors:
            # world_from_levelled = SOP @ scanner_from_levelled
            priors = [None if p.sop is None
                      else p.sop @ (np.eye(4) if lv is None else np.linalg.inv(lv))
                      for p, lv in zip(positions, levelling, strict=True)]
    result = coreg.coregister(clouds, cfg, names=names, levelling=levelling, reflectors=reflectors,
                              approximate_positions=gnss, priors=priors)
    result.save(out / "transforms.json")
    (out / "report.txt").write_text(result.report() + "\n")
    for k, name in enumerate(result.names):
        if result.registered[k]:
            np.savetxt(out / f"{name}.dat", result.transform_for(k), fmt="%.12f")
    msg = f"{sum(result.registered)} of {len(names)} scans registered -> {out}"
    if args.merged:
        cloud = coreg.merge_clouds(clouds, result, voxel=args.voxel,
                                   riegl_options=cfg.riegl_options,
                                   riscan_filter=cfg.riscan_filter)
        io.write(cloud, args.merged)
        msg += f"; merged {len(cloud):,} points -> {args.merged}"
    print(msg)


def _write_raster(r: Raster, path: str) -> None:
    if path.lower().endswith((".tif", ".tiff")):
        r.to_geotiff(path)
    else:
        r.to_ascii_grid(path)


def _als_run(args) -> dict:
    return {"chunk_size": args.chunk_size, "buffer": args.buffer, "workers": args.workers}


def _cmd_als_catalog(args):
    cat = als.catalog(args.input, pattern=args.pattern, recursive=args.recursive,
                      tolerance=args.tolerance)
    print(cat.report(), end="")
    if args.strict and cat.issues():
        raise ValueError(f"{len(cat.issues())} problem(s) in the catalogue")


def _cmd_als_ground(args):
    cat = als.catalog(args.input, pattern=args.pattern)
    out = als.classify_ground(cat, args.output, method=args.method,
                              cloth_resolution=args.resolution, cell_size=args.resolution,
                              last_returns=args.last_returns, **_als_run(args))
    print(f"classified {out.n_points:,} points in {len(out)} tiles -> {args.output}")


def _cmd_als_dtm(args):
    cat = als.catalog(args.input, pattern=args.pattern)
    dtm = als.dtm(cat, resolution=args.resolution, method=args.method, **_als_run(args))
    _write_raster(dtm, args.output)
    print(f"DTM {dtm.shape} at {args.resolution} m from {len(cat)} tiles -> {args.output}")


def _cmd_als_chm(args):
    cat = als.catalog(args.input, pattern=args.pattern)
    dtm = None if args.normalized else (Raster.from_ascii_grid(args.dtm) if args.dtm else "auto")
    chm = als.chm(cat, resolution=args.resolution, dtm=dtm, dtm_resolution=args.dtm_resolution,
                  min_height=args.min_height, **_als_run(args))
    _write_raster(chm, args.output)
    print(f"CHM {chm.shape} at {args.resolution} m, max {np.nanmax(chm.data):.1f} m -> "
          f"{args.output}")


def _cmd_als_normalize(args):
    cat = als.catalog(args.input, pattern=args.pattern)
    dtm = Raster.from_ascii_grid(args.dtm) if args.dtm else "auto"
    out = als.normalize(cat, args.output, dtm=dtm, dtm_resolution=args.dtm_resolution,
                        replace_z=args.replace_z, **_als_run(args))
    print(f"normalised {out.n_points:,} points in {len(out)} tiles -> {args.output}")


def _cmd_als_trees(args):
    cat = als.catalog(args.input, pattern=args.pattern)
    dtm = None if args.normalized else (Raster.from_ascii_grid(args.dtm) if args.dtm else "auto")
    if args.window_linear is not None:
        window = als.LinearWindow(*args.window_linear)
    else:
        window = args.window
    trees = als.find_trees(
        cat, out=args.labelled, method=args.method, resolution=args.resolution, dtm=dtm,
        dtm_resolution=args.dtm_resolution, window=window, hmin=args.hmin, shape=args.shape,
        tops_from=args.tops_from, th_tree=args.th_tree, th_seed=args.th_seed, th_cr=args.th_cr,
        max_cr=args.max_cr, dt1=args.dt1, dt2=args.dt2, R=args.R, Zu=args.Zu,
        speed_up=args.speed_up, hull=args.hull, concavity=args.concavity, smooth=args.smooth,
        min_point_height=args.min_point_height, **_als_run(args))
    trees.to_csv(args.output)
    msg = f"{len(trees):,} trees from {len(cat)} tiles -> {args.output}"
    if args.crowns:
        trees.to_geojson(args.crowns, geometry="tops" if args.method == "tops" else "crowns")
        msg += f"; {'tops' if args.method == 'tops' else 'crowns'} -> {args.crowns}"
    if args.labelled:
        msg += f"; labelled tiles -> {args.labelled}"
    print(msg)


def _als_common(s, buffer: float = 20.0):
    s.add_argument("--pattern", default="*.la[sz]", help="file name pattern within the directory")
    s.add_argument("--chunk-size", type=float, default=None,
                   help="process square chunks of this size (m) rather than one tile at a time")
    s.add_argument("--buffer", type=float, default=buffer,
                   help="band of neighbouring points read around each chunk (m)")
    s.add_argument("--workers", type=int, default=None,
                   help="chunks at once (default: one per CPU, fewer if memory is short)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sylva",
                                description="Terrestrial laser scanning for forest ecology.",
                                epilog="Run `sylva <command> --help` for the options of a command.")
    p.add_argument("--version", action="version", version=f"sylva {__version__}")
    p.add_argument("--no-progress", action="store_true",
                   help="do not draw the progress bar (it is drawn on a terminal by default)")
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

    s = sub.add_parser("reproject", help="reproject a point cloud into another CRS", **fmt)
    s.add_argument("input", help="point cloud")
    s.add_argument("output", help="point cloud; format from the extension (LAS/LAZ store the CRS)")
    s.add_argument("--to", required=True, metavar="CRS",
                   help="target CRS: EPSG code (EPSG:7855), PROJ string or WKT")
    s.add_argument("--from", dest="src", metavar="CRS",
                   help="source CRS (default: the CRS in the input's LAS header)")
    s.set_defaults(func=_cmd_reproject)

    s = sub.add_parser("transform", help="apply a 4x4 matrix, a shift or a rotation", **fmt)
    s.add_argument("input", help="point cloud")
    s.add_argument("output", help="point cloud; format from the extension")
    how = s.add_mutually_exclusive_group(required=True)
    how.add_argument("--matrix", metavar="FILE",
                     help="4x4 matrix file (16 numbers, row-major; RiSCAN .DAT, sylva coreg .dat)")
    how.add_argument("--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                     help="shift by this offset (m)")
    how.add_argument("--rotate", type=float, metavar="DEG",
                     help="rotate by this angle (degrees, counter-clockwise seen from +axis)")
    s.add_argument("--axis", choices=["x", "y", "z"], default="z", help="rotation axis")
    s.add_argument("--about", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="point the rotation axis passes through (default: the origin)")
    s.set_defaults(func=_cmd_transform)

    s = sub.add_parser("ground", help="classify ground, build DTM and add height attribute", **fmt)
    s.add_argument("input", help="point cloud, z up")
    s.add_argument("output", nargs="?", default=None,
                   help="cloud with classification (2 = ground) and height attributes "
                        "(default: <input>_norm beside the input)")
    s.add_argument("--method", choices=["csf", "pmf"], default="csf",
                   help="cloth simulation or progressive morphological filter")
    s.add_argument("--resolution", type=float, default=0.5,
                   help="cloth / filter cell and DTM resolution (m)")
    s.add_argument("--dtm", help="also write the DTM (.tif needs rasterio, else .asc)")
    s.set_defaults(func=_cmd_ground)

    s = sub.add_parser("trees", help="detect stems and DBH from a height-normalised cloud", **fmt)
    s.add_argument("input", help="cloud with a height attribute (from `sylva ground`)")
    s.add_argument("-o", "--output",
                   help="CSV of trees, one row each with crown metrics "
                        "(default: <input>_trees.csv beside the input; - for stdout)")
    s.add_argument("--min-dbh", type=float, default=0.05,
                   help="smallest stem diameter detected (m)")
    s.add_argument("--min-height", type=float, default=3.0,
                   help="drop candidates whose segment is lower than this (m)")
    s.add_argument("--segment", nargs="?", const="", default=None, metavar="PATH",
                   help="write a cloud with a tree_id attribute (bare flag: <input>_segmented "
                        "beside the input)")
    s.set_defaults(func=_cmd_trees)

    s = sub.add_parser("chm", help="canopy height model from a height-normalised cloud", **fmt)
    s.add_argument("input", help="cloud with a height attribute")
    s.add_argument("output", nargs="?", default=None,
                   help=".tif (needs rasterio) or .asc (default: <input>_chm.asc beside the input)")
    s.add_argument("--resolution", type=float, default=0.5, help="cell size (m)")
    s.set_defaults(func=_cmd_chm)

    s = sub.add_parser("pad", help="plant area density profile (voxel method), CSV on stdout",
                       **fmt)
    s.add_argument("input", help="cloud with a height attribute")
    s.add_argument("--voxel", type=float, default=0.5, help="voxel size and layer thickness (m)")
    s.set_defaults(func=_cmd_pad)

    s = sub.add_parser("qsm", help="build a cylinder model of a single tree", **fmt)
    s.add_argument("input", help="one tree's wood points")
    s.add_argument("output", nargs="?", default=None,
                   help="CSV of cylinders (default: <input>_qsm.csv beside the input)")
    s.add_argument("--bin-length", type=float, default=0.3, help="geodesic shell width (m)")
    s.set_defaults(func=_cmd_qsm)

    s = sub.add_parser("qsm-plot", help="build a QSM for every tree of a segmented cloud", **fmt)
    s.add_argument("input", help="height-normalised cloud with a tree id attribute")
    s.add_argument("output", nargs="?", default=None,
                   help="CSV, one row per tree (default: <input>_trees.csv beside the input)")
    s.add_argument("--tree-attr", default="tree_id", help="attribute holding the tree id")
    s.add_argument("--cylinders", nargs="?", const="", default=None, metavar="DIR",
                   help="also write one cylinder CSV per tree (bare flag: <input>_cylinders "
                        "beside the input)")
    s.add_argument("--meshes", nargs="?", const="", default=None, metavar="DIR",
                   help="also write a surface mesh per tree (bare flag: <input>_meshes "
                        "beside the input)")
    s.add_argument("--mesh-format", choices=("ply", "obj"), default="ply",
                   help="format for --meshes")
    s.add_argument("--voxel", type=float, default=0.01, help="thin each tree to this spacing (m)")
    s.add_argument("--bin-length", type=float, default=0.1, help="geodesic shell width (m)")
    s.add_argument("--min-points", type=int, default=2000, help="skip trees with fewer points")
    s.add_argument("--buttress", action="store_true",
                   help="mesh a buttressed base and count it in the volume")
    s.add_argument("--no-wood", action="store_true",
                   help="skip the leaf/wood filter (the cloud is wood already)")
    s.set_defaults(func=_cmd_qsm_plot)

    s = sub.add_parser("shots", help="convert a ray cloud to a sylva shots file (.parquet)", **fmt)
    s.add_argument("input", help="ray cloud with sx,sy,sz or nx,ny,nz attributes")
    s.add_argument("output", nargs="?", default=None,
                   help="shots file (default: <input>.parquet beside the input)")
    s.add_argument("--double", action="store_true", help="double-precision angles and ranges")
    s.set_defaults(func=_cmd_shots)

    s = sub.add_parser("voxel", help="ray-traced voxel grid (AMAPVox-style) from pulse data", **fmt)
    s.add_argument("input", help="shots file (.parquet, streamed) or ray cloud")
    s.add_argument("output", nargs="?", default=None,
                   help=".vox (AMAPVox) or .txt (default: <input>.vox beside the input)")
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

    s = sub.add_parser("coreg", help="marker-free coregistration of scan positions",
                       **fmt)
    s.add_argument("inputs", nargs="+",
                   help="a RiSCAN / scanner .PROJ project directory, or scan files "
                        "(.rxp, .laz, ...)")
    s.add_argument("-o", "--output", help="output folder (default: <project>_coreg beside it)")
    s.add_argument("--reference", help="scan name or index whose frame is the world frame")
    s.add_argument("--level", choices=["auto", "attitude", "sop", "none"], default="auto",
                   help="level tilted scans with the scanner's attitude, the SOP rotation, or not "
                        "(auto: attitude, else SOP)")
    s.add_argument("--sop-priors", action="store_true",
                   help="use the project's SOPs as priors: refuse results far from them and place "
                        "scans with too few stems from them")
    s.add_argument("--refine", action="store_true", help="joint multi-view refinement of all poses")
    s.add_argument("--no-reflectors", action="store_true", help="do not use reflective targets")
    s.add_argument("--max-pair-distance", type=float, default=40.0,
                   help="skip pairs further apart by GNSS (m)")
    s.add_argument("--min-range", type=float, help="drop echoes closer to the scanner (m)")
    s.add_argument("--max-range", type=float, help="drop echoes further from the scanner (m)")
    s.add_argument("--min-deviation", type=float)
    s.add_argument("--max-deviation", type=float, help="drop echoes with a larger pulse deviation")
    s.add_argument("--min-reflectance", type=float)
    s.add_argument("--max-reflectance", type=float)
    s.add_argument("--min-amplitude", type=float)
    s.add_argument("--max-amplitude", type=float)
    s.add_argument("--riscan-export-settings", metavar="FILE",
                   help="range/deviation/reflectance/amplitude intervals from a RiSCAN PRO export "
                        "filter settings file ('attribute, min, max' per line); the bounds above "
                        "override it")
    s.add_argument("--riscan-filter", choices=["none", "current", "legacy"], default="none",
                   help="RiSCAN PRO's RXP import filter: 'current' drops echoes within 0.5 m of "
                        "the scanner, 'legacy' also the weak isolated echoes the older conversion "
                        "discarded")
    s.add_argument("--trust-reflectors", type=int, metavar="N",
                   help="accept reflector matches of at least N targets (within 3 cm) even when "
                        "ICP fails; 0 always asks ICP to agree (default 5)")
    s.add_argument("--workers", type=int, default=0, help="scans and pairs at once (0: automatic)")
    s.add_argument("--merged", help="also write the merged, registered cloud here")
    s.add_argument("--voxel", type=float, default=0.02, help="thinning of the merged cloud (m)")
    s.add_argument("--quiet", action="store_true", help="only print the summary")
    s.set_defaults(func=_cmd_coreg)

    s = sub.add_parser("als-catalog", help="summarise and check a directory of ALS tiles "
                       "(headers only)", **fmt)
    s.add_argument("input", nargs="+", help="directory of LAS/LAZ tiles, or files")
    s.add_argument("--pattern", default="*.la[sz]", help="file name pattern within directories")
    s.add_argument("--recursive", action="store_true", help="search subdirectories too")
    s.add_argument("--tolerance", type=float, default=1.0,
                   help="overlap or shortfall between tile extents that counts as a problem (m)")
    s.add_argument("--strict", action="store_true", help="exit with status 1 if problems are found")
    s.set_defaults(func=_cmd_als_catalog)

    s = sub.add_parser("als-ground", help="classify ground over a directory of ALS tiles", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles")
    s.add_argument("output", help="directory for the classified tiles")
    s.add_argument("--method", choices=["csf", "pmf"], default="csf",
                   help="cloth simulation or progressive morphological filter")
    s.add_argument("--resolution", type=float, default=0.5, help="cloth or filter cell size (m)")
    s.add_argument("--last-returns", action="store_true", help="only last returns can be ground")
    _als_common(s)
    s.set_defaults(func=_cmd_als_ground)

    s = sub.add_parser("als-dtm", help="DTM of a directory of ground-classified ALS tiles", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles with ground classified")
    s.add_argument("output", help=".asc, or .tif (needs rasterio)")
    s.add_argument("--resolution", type=float, default=1.0, help="cell size (m)")
    s.add_argument("--method", choices=["lowest", "tin", "natural", "idw"], default="lowest",
                   help="lowest ground point per cell, or an interpolation at cell centres")
    _als_common(s)
    s.set_defaults(func=_cmd_als_dtm)

    s = sub.add_parser("als-chm", help="canopy height model of a directory of ALS tiles", **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles with ground classified")
    s.add_argument("output", help=".asc, or .tif (needs rasterio)")
    s.add_argument("--resolution", type=float, default=0.5, help="cell size (m)")
    s.add_argument("--dtm-resolution", type=float, default=1.0,
                   help="cell size of the DTM made on the fly from the ground points (m)")
    s.add_argument("--dtm", help="use this DTM (.asc) instead of making one")
    s.add_argument("--normalized", action="store_true",
                   help="the tiles are already normalised (z is height)")
    s.add_argument("--min-height", type=float, default=0.0,
                   help="cells with nothing this high are 0 (m)")
    _als_common(s)
    s.set_defaults(func=_cmd_als_chm)

    s = sub.add_parser("als-normalize", help="height above ground for a directory of ALS tiles",
                       **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles with ground classified")
    s.add_argument("output", help="directory for the normalised tiles")
    s.add_argument("--dtm-resolution", type=float, default=1.0,
                   help="cell size of the DTM made on the fly from the ground points (m)")
    s.add_argument("--dtm", help="use this DTM (.asc) instead of making one")
    s.add_argument("--replace-z", action="store_true",
                   help="replace z by the height (elevation kept as an attribute) rather than "
                        "adding a height attribute")
    _als_common(s)
    s.set_defaults(func=_cmd_als_normalize)

    from . import als_metrics
    als_metrics._add_commands(sub, fmt, _als_common, _write_raster)
    from .change import als as change_als
    change_als._add_commands(sub, fmt, _als_common, _write_raster)

    from . import fusion
    fusion._add_commands(sub, fmt)

    from . import tiles
    tiles._add_commands(sub, fmt)

    s = sub.add_parser("als-trees", help="tree tops and crowns over a directory of ALS tiles",
                       **fmt)
    s.add_argument("input", help="directory of LAS/LAZ tiles (ground classified, or normalised)")
    s.add_argument("output", help="CSV of the trees: id, x, y, height, crown_area, n_points")
    s.add_argument("--crowns", help="also write the crown polygons (the tops with --method tops) "
                                    "as GeoJSON")
    s.add_argument("--labelled", help="also write the tiles here with each point's tree id")
    s.add_argument("--method", choices=["dalponte2016", "watershed", "li2012", "tops"],
                   default="dalponte2016", help="crown segmentation, or tree tops only")
    s.add_argument("--resolution", type=float, default=0.5, help="CHM cell size (m)")
    s.add_argument("--window", type=float, default=5.0,
                   help="local maximum window diameter (m)")
    s.add_argument("--window-linear", type=float, nargs=4,
                   metavar=("INTERCEPT", "SLOPE", "MIN", "MAX"),
                   help="window growing with height h: clip(INTERCEPT + SLOPE h, MIN, MAX)")
    s.add_argument("--hmin", type=float, default=2.0, help="lowest tree top (m)")
    s.add_argument("--shape", choices=["circular", "square"], default="circular",
                   help="window shape")
    s.add_argument("--tops-from", choices=["chm", "points"], default="chm",
                   help="find tops on the CHM or on the points")
    s.add_argument("--smooth", type=int, default=0,
                   help="mean-filter the CHM over (2k+1)^2 cells first")
    s.add_argument("--th-tree", type=float, default=2.0, help="lowest crown cell (m)")
    s.add_argument("--th-seed", type=float, default=0.45, help="dalponte2016 seed threshold")
    s.add_argument("--th-cr", type=float, default=0.55, help="dalponte2016 crown threshold")
    s.add_argument("--max-cr", type=float, default=10.0,
                   help="dalponte2016 crown extent from the top (cells)")
    s.add_argument("--dt1", type=float, default=1.5, help="li2012 spacing below Zu (m)")
    s.add_argument("--dt2", type=float, default=2.0, help="li2012 spacing above Zu (m)")
    s.add_argument("--R", type=float, default=2.0, help="li2012 local maximum window (m)")
    s.add_argument("--Zu", type=float, default=15.0, help="li2012 height switching dt1 to dt2 (m)")
    s.add_argument("--speed-up", type=float, default=10.0, help="li2012 largest crown radius (m)")
    s.add_argument("--hull", choices=["convex", "concave"], default="convex",
                   help="crown outline")
    s.add_argument("--concavity", type=float, default=2.0,
                   help="edge length of the concave outline (m)")
    s.add_argument("--min-point-height", type=float, default=0.5,
                   help="points lower than this belong to no tree (m)")
    s.add_argument("--dtm-resolution", type=float, default=1.0,
                   help="cell size of the DTM made on the fly from the ground points (m)")
    s.add_argument("--dtm", help="use this DTM (.asc) instead of making one")
    s.add_argument("--normalized", action="store_true",
                   help="the tiles are already normalised (z is height)")
    _als_common(s, buffer=30.0)
    s.set_defaults(func=_cmd_als_trees)

    args = p.parse_args(argv)
    try:
        if args.no_progress:
            args.func(args)
        else:
            with progress.bar():
                args.func(args)
    except (OSError, ValueError, KeyError, ImportError) as e:
        p.exit(1, f"sylva: error: {e}\n")


if __name__ == "__main__":
    main()
