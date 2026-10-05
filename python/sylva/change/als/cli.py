# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Command-line interface of the airborne change functions."""

from __future__ import annotations

from pathlib import Path

from ...raster import Raster
from .align import align_surveys
from .gaps import gap_change
from .surface import chm_change, harmonise, surface_change
from .trees import tree_change


def _cmd_als_change(args, write_raster):
    from ... import als
    a = als.catalog(args.a, pattern=args.pattern)
    b = als.catalog(args.b, pattern=args.pattern)
    run = {"chunk_size": args.chunk_size, "buffer": args.buffer, "workers": args.workers}
    al = None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.align:
        classes = [int(c) for c in args.stable_classes.split(",")]
        al = align_surveys(a, b, block_size=args.block_size, stable_classes=classes,
                           model=args.model, workers=args.workers)
        al.to_csv(out / "alignment.csv")
        print(al.report())
    ch = surface_change(a, b, args.surface, args.resolution, alignment=al,
                        harmonise=args.harmonise, density_cell=args.density_cell,
                        first_returns=args.first_returns, confidence=args.confidence, **run)
    ext = "tif" if args.format == "tif" else "asc"
    for name, r in (("a", ch.a), ("b", ch.b), ("difference", ch.difference), ("lod", ch.lod)):
        write_raster(r, str(out / f"{args.surface}_{name}.{ext}"))
    write_raster(Raster(ch.classes.astype(float), ch.a.xmin, ch.a.ymin, ch.a.resolution, ch.a.crs),
                 str(out / f"{args.surface}_classes.{ext}"))
    print(ch.report())
    if args.gaps and args.surface == "chm":
        g = gap_change(ch, height=args.gap_height, min_area=args.gap_min_area, years=args.years)
        g.to_geojson(out / "gaps_a.geojson", "a")
        g.to_geojson(out / "gaps_b.geojson", "b")
        print(g.report())
    print(f"-> {out}")


def _cmd_als_tree_change(args, write_raster):
    import tempfile

    from ... import als
    a = als.catalog(args.a, pattern=args.pattern)
    b = als.catalog(args.b, pattern=args.pattern)
    run = {"chunk_size": args.chunk_size, "buffer": args.buffer, "workers": args.workers}
    al = None
    if args.align:
        classes = [int(c) for c in args.stable_classes.split(",")]
        al = align_surveys(a, b, block_size=args.block_size, stable_classes=classes,
                           workers=args.workers)
        print(al.report())
    window = als.LinearWindow(*args.window_linear) if args.window_linear else args.window
    kw = dict(method=args.method, resolution=args.resolution, window=window, hmin=args.hmin,
              max_cr=args.max_cr, **run)
    with tempfile.TemporaryDirectory() as tmp:
        ta_cat, tb_cat = a, b
        if args.harmonise:
            ta_cat, tb_cat = harmonise(a, b, Path(tmp) / "a", Path(tmp) / "b",
                                       density_cell=args.density_cell, workers=args.workers)
        ta = als.find_trees(ta_cat, **kw)
        tb = als.find_trees(tb_cat, **kw)
    ch = chm_change(a, b, args.resolution, alignment=al, harmonise=args.harmonise,
                    density_cell=args.density_cell, confidence=args.confidence, **run)
    tc = tree_change(ta, tb, ch, alignment=al, max_distance=args.max_distance,
                     confidence=args.confidence)
    tc.to_csv(args.output)
    print(tc.report(years=args.years))
    if args.grid:
        g = tc.grid(args.grid_resolution)
        d = Path(args.grid)
        d.mkdir(parents=True, exist_ok=True)
        for k, r in g.items():
            write_raster(r, str(d / f"{k}.asc"))
    print(f"{len(ta)} and {len(tb)} trees compared -> {args.output}")


def _add_commands(sub, fmt, common, write_raster) -> None:
    """Add ``als-change`` and ``als-tree-change`` to the command line."""
    s = sub.add_parser("als-change", help="CHM, DSM or DTM change between two ALS surveys, "
                       "with a level of detection per cell and gap dynamics", **fmt)
    s.add_argument("a", help="directory of the earlier survey's tiles (ground classified)")
    s.add_argument("b", help="directory of the later survey's tiles (ground classified)")
    s.add_argument("output", help="directory for the rasters, gap polygons and alignment")
    s.add_argument("--surface", choices=["chm", "dsm", "dtm"], default="chm",
                   help="what to compare")
    s.add_argument("--resolution", type=float, default=1.0, help="cell size (m)")
    s.add_argument("--align", action="store_true",
                   help="estimate the offsets between the surveys on stable surfaces first")
    s.add_argument("--stable-classes", default="2",
                   help="comma-separated classes of stable returns (2 ground, 6 roofs, 11 roads)")
    s.add_argument("--block-size", type=float, default=100.0, help="alignment block size (m)")
    s.add_argument("--model", choices=["field", "blocks", "constant"], default="field",
                   help="alignment model")
    s.add_argument("--harmonise", action="store_true",
                   help="thin the denser survey to the other's pulse density")
    s.add_argument("--density-cell", type=float, default=10.0,
                   help="square in which pulse densities are compared (m)")
    s.add_argument("--first-returns", action="store_true", help="grid first returns only")
    s.add_argument("--confidence", type=float, default=0.95, help="of the level of detection")
    s.add_argument("--gaps", action="store_true", help="also find gaps and their change (CHM)")
    s.add_argument("--gap-height", type=float, default=2.0, help="gap height threshold (m)")
    s.add_argument("--gap-min-area", type=float, default=10.0, help="smallest gap (m²)")
    s.add_argument("--years", type=float, default=None, help="time between the surveys")
    s.add_argument("--format", choices=["asc", "tif"], default="asc",
                   help="raster format")
    common(s)
    s.set_defaults(func=lambda args: _cmd_als_change(args, write_raster))

    s = sub.add_parser("als-tree-change", help="trees of two ALS surveys matched: growth, "
                       "mortality, damage and recruitment", **fmt)
    s.add_argument("a", help="directory of the earlier survey's tiles (ground classified)")
    s.add_argument("b", help="directory of the later survey's tiles (ground classified)")
    s.add_argument("output", help="CSV with one row per tree")
    s.add_argument("--resolution", type=float, default=0.5, help="CHM cell size (m)")
    s.add_argument("--method", choices=["dalponte2016", "watershed", "li2012"],
                   default="dalponte2016", help="crown segmentation")
    s.add_argument("--window", type=float, default=5.0, help="local maximum window diameter (m)")
    s.add_argument("--window-linear", type=float, nargs=4,
                   metavar=("INTERCEPT", "SLOPE", "MIN", "MAX"),
                   help="window growing with height h: clip(INTERCEPT + SLOPE h, MIN, MAX)")
    s.add_argument("--hmin", type=float, default=2.0, help="lowest tree top (m)")
    s.add_argument("--max-cr", type=float, default=10.0,
                   help="dalponte2016 crown extent from the top (cells)")
    s.add_argument("--max-distance", type=float, default=1.5,
                   help="farthest apart the tops of one tree can be (m)")
    s.add_argument("--align", action="store_true", help="estimate the offsets first")
    s.add_argument("--stable-classes", default="2", help="comma-separated stable classes")
    s.add_argument("--block-size", type=float, default=100.0, help="alignment block size (m)")
    s.add_argument("--harmonise", action="store_true",
                   help="thin the denser survey to the other's pulse density before comparing")
    s.add_argument("--density-cell", type=float, default=10.0,
                   help="square in which pulse densities are compared (m)")
    s.add_argument("--confidence", type=float, default=0.95, help="of the levels of detection")
    s.add_argument("--years", type=float, default=None, help="time between the surveys")
    s.add_argument("--grid", help="also write per-cell totals as rasters into this directory")
    s.add_argument("--grid-resolution", type=float, default=50.0, help="cell size of --grid (m)")
    common(s, buffer=30.0)
    s.set_defaults(func=lambda args: _cmd_als_tree_change(args, write_raster))
