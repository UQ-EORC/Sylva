# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Command-line interface of the fusion functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..raster import Raster
from .link import link_trees
from .registration import Registration, register
from .upscaling import upscale


def _add_commands(sub, fmt: dict) -> None:
    """The ``fusion-register``, ``fusion-trees`` and ``fusion-upscale`` commands."""
    s = sub.add_parser("fusion-register", help="register a TLS plot onto ALS tiles or an ALS cloud", **fmt)
    s.add_argument("tls", help="TLS point cloud with ground classified (class 2)")
    s.add_argument("als", help="ALS directory of tiles, or one file, with ground classified")
    s.add_argument("output", help="JSON file for the registration (transform, residuals, uncertainty)")
    s.add_argument("--initial", help="text file with the 4x4 matrix placing the TLS roughly on the ALS")
    s.add_argument("--transformed", help="also write the TLS cloud moved into the ALS frame here")
    s.add_argument("--search-radius", type=float, default=10.0, help="largest horizontal shift searched (m)")
    s.add_argument("--heading-range", type=float, default=180.0,
                   help="headings searched either side of the initial one (degrees; 180 for all)")
    s.add_argument("--heading-step", type=float, default=3.0, help="coarse heading step (degrees)")
    s.add_argument("--resolution", type=float, default=0.5, help="fine search cell size (m)")
    s.add_argument("--coarse-resolution", type=float, default=2.0, help="coarse search cell size (m)")
    s.add_argument("--dtm-weight", type=float, default=0.5, help="weight of the terrain correlation")
    s.add_argument("--refine", choices=["all", "ground", "none"], default="all",
                   help="points the ICP refinement uses")
    s.add_argument("--no-jackknife", action="store_true", help="skip the quadrant jackknife")
    s.set_defaults(func=_cmd_register)

    s = sub.add_parser("fusion-trees", help="link TLS trees to ALS trees and combine DBH and height", **fmt)
    s.add_argument("tls_trees", help="CSV of TLS trees: x, y and optionally tree_id, dbh, height, volume")
    s.add_argument("als_trees", help="CSV of ALS trees (from als-trees): id, x, y, height")
    s.add_argument("output", help="CSV of the linked TLS trees")
    s.add_argument("--crowns", help="GeoJSON of the ALS crowns (from als-trees --crowns)")
    s.add_argument("--registration", help="JSON from fusion-register, applied to the TLS positions")
    s.add_argument("--als-output", help="also write the ALS trees with their stems as CSV")
    s.add_argument("--max-distance", type=float, default=3.0, help="stem to top distance allowed (m)")
    s.add_argument("--crown-buffer", type=float, default=0.5, help="distance outside a crown still under it (m)")
    s.add_argument("--height-weight", type=float, default=2.0, help="weight of the height difference")
    s.add_argument("--dbh-weight", type=float, default=2.0, help="weight of the DBH shortfall")
    s.add_argument("--max-cost", type=float, default=2.0, help="cost of leaving a stem unmatched")
    s.add_argument("--top-tolerance", type=float, default=1.0, help="height within which a top is reached (m)")
    s.set_defaults(func=_cmd_trees)

    s = sub.add_parser("fusion-upscale", help="regress plot values on ALS metrics and predict them "
                       "wall to wall", **fmt)
    s.add_argument("plots", help="CSV with one row per plot: the value and the ALS metrics (e.g. "
                   "als-plot-metrics output with a column added)")
    s.add_argument("metrics", help="directory of metric rasters <name>.asc (from als-metrics)")
    s.add_argument("output", help="directory for mean.asc, se.asc, lower.asc, upper.asc, extrapolated.asc")
    s.add_argument("--response", required=True, help="column of the plot value (e.g. agb)")
    s.add_argument("--predictors", required=True, help="comma-separated metric names")
    s.add_argument("--model", choices=["loglog", "linear"], default="loglog", help="model form")
    s.add_argument("--level", type=float, default=0.95, help="coverage of the prediction interval")
    s.set_defaults(func=_cmd_upscale)


def _read_csv(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[k] = np.array(vals)
    return out


def _cmd_register(args) -> None:
    from ..io import read, write

    tls = read(args.tls)
    init = None if args.initial is None else np.loadtxt(args.initial).reshape(4, 4)
    als = args.als if Path(args.als).is_dir() else read(args.als)
    reg = register(tls, als, init, resolution=args.resolution, coarse_resolution=args.coarse_resolution,
                   search_radius=args.search_radius, heading_range=args.heading_range,
                   heading_step=args.heading_step, dtm_weight=args.dtm_weight, refine=args.refine,
                   jackknife=not args.no_jackknife)
    reg.save(args.output)
    print(reg.report())
    if args.transformed:
        write(reg.apply(tls), args.transformed)


def _cmd_trees(args) -> None:
    tls = _read_csv(args.tls_trees)
    als = _read_csv(args.als_trees)
    if args.crowns:
        gj = json.loads(Path(args.crowns).read_text())
        by_id = {}
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Polygon":
                ring = np.asarray(geom["coordinates"][0], dtype=float)
                by_id[int(feat["properties"]["id"])] = ring[:-1] if len(ring) > 1 and np.allclose(ring[0], ring[-1]) else ring
        ids = als["id"].astype(int) if "id" in als else np.arange(1, len(als["x"]) + 1)
        als["crowns"] = [by_id.get(int(i), np.zeros((0, 2))) for i in ids]
    reg = None if args.registration is None else Registration.load(args.registration)
    links = link_trees(tls, als, reg, max_distance=args.max_distance, crown_buffer=args.crown_buffer,
                       height_weight=args.height_weight, dbh_weight=args.dbh_weight,
                       max_cost=args.max_cost, top_tolerance=args.top_tolerance)
    links.to_csv(args.output)
    if args.als_output:
        t = links.als_table()
        with open(args.als_output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "y", "height", "tls_id", "n_stems", "stems"])
            for k in range(len(t["id"])):
                w.writerow([t["id"][k].item() if hasattr(t["id"][k], "item") else t["id"][k], float(t["x"][k]),
                            float(t["y"][k]), float(t["height"][k]), int(t["tls_id"][k]), int(t["n_stems"][k]),
                            " ".join(str(int(v)) for v in t["stems"][k])])
    print(links)


def _cmd_upscale(args) -> None:
    plots = _read_csv(args.plots)
    if args.response not in plots:
        raise ValueError(f"{args.plots} has no column {args.response!r}")
    names = [n.strip() for n in args.predictors.split(",") if n.strip()]
    grid = {}
    for n in names:
        p = Path(args.metrics) / f"{n}.asc"
        if not p.exists():
            raise ValueError(f"no raster {p}")
        grid[n] = Raster.from_ascii_grid(p)
    up = upscale(plots[args.response], plots, grid, names, args.model, args.level)
    up.to_ascii_grids(args.output)
    print(up.model.summary())
