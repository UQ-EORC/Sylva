"""Build Sylva QSMs for the destructive-harvest reference trees.

Runs on the isolated single-tree clouds in
``$SYLVA_HARVEST/data/harvest/ply_ref`` (the same inputs the other methods
were run on). The clouds hold no ground and start at z = 0 at the stem
base, so height above ground is z itself. Writes one cylinder CSV per tree
to ``out_pytls_ref/<cloud>.csv`` and a summary ``pytls_metrics.csv``.

Usage: python benchmarks/harvest_qsm.py [--only CLOUD ...] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path

import numpy as np

import sylva
from sylva import filters, qsm, trees
from paths import HARVEST_DATA

ROOT = HARVEST_DATA


def base_and_dbh(tree: sylva.PointCloud) -> tuple[tuple[float, float], float]:
    """Stem base position and a slice-based DBH (multi-slice detector on the
    tree alone; falls back to a 1.3 m RANSAC circle, then to NaN)."""
    z = tree.z - tree.z.min()
    t = tree.with_attrs(height=z)
    found = trees.detect_stems(t, min_arc_deg=90.0)
    if found:
        best = max(found, key=lambda s: s.n_points)
        return (best.x, best.y), best.dbh
    sel = (z > 1.2) & (z < 1.4)
    if sel.sum() >= 10:
        try:
            cx, cy, r, _ = trees.fit_circle_ransac(tree.xyz[sel, :2], threshold=0.02,
                                                   max_radius=1.5)
            return (cx, cy), 2 * r
        except ValueError:
            pass
    low = tree.xyz[z < 0.5]
    return (float(np.median(low[:, 0])), float(np.median(low[:, 1]))), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--params", default="{}", help="extra build_qsm keyword arguments as a Python dict literal")
    ap.add_argument("--mesh", action="store_true", help="also write OBJ and PLY meshes to out_pytls_ref/mesh")
    args = ap.parse_args()
    extra = dict(eval(args.params))
    out = ROOT / "out_pytls_ref"
    out.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(ROOT / "ply_ref" / "*.ply")))
    if args.only:
        files = [f for f in files if Path(f).stem in args.only]
    if args.limit:
        files = files[: args.limit]
    rows = []
    for f in files:
        name = Path(f).stem
        t0 = time.time()
        pc = sylva.read(f).without("nx", "ny", "nz", "time", "red", "green", "blue", "alpha")
        thin = filters.voxel_downsample(pc, 0.01)
        (bx, by), dbh = base_and_dbh(thin)
        wood = qsm.wood_points(thin, k=20, threshold=0.85, voxel_size=0.02)
        row = {"cloud": name, "n_points": len(pc), "n_wood": len(wood), "dbh_slice_m": dbh,
               "h_cloud": float(pc.z.max() - pc.z.min())}
        try:
            base_r = dbh / 2 if np.isfinite(dbh) else 0.0
            max_r = 1.5  # do not cap on the slice DBH: it fails on buttressed stems
            model = qsm.build_qsm(wood, base_xy=(bx, by), max_radius=max_r, base_radius=base_r, **extra)
            model.to_csv(out / f"{name}.csv")
            if args.mesh:
                (out / "mesh").mkdir(exist_ok=True)
                model.to_obj(out / "mesh" / f"{name}.obj")
                model.to_ply(out / "mesh" / f"{name}.ply")
            s = model.summary()
            row.update(status="ok", V=s["total_volume_m3"], V_stem=s["stem_volume_m3"],
                       DBH=s["dbh_m"], H=float(model.end[:, 2].max() - model.start[:, 2].min()),
                       n_cyl=s["n_cylinders"], orders=s["max_branch_order"],
                       length=s["total_length_m"])
        except Exception as exc:  # noqa: BLE001 - record and continue
            row.update(status=f"error: {exc}")
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"[{name}] {row.get('status')} V {row.get('V', float('nan')):.3f} DBH {row.get('DBH', float('nan')):.3f} "
              f"(slice {dbh:.3f}) H {row.get('H', float('nan')):.1f}/{row['h_cloud']:.1f} cyl {row.get('n_cyl', 0)} {row['seconds']}s", flush=True)
    keys = ["cloud", "status", "n_points", "n_wood", "dbh_slice_m", "h_cloud", "V", "V_stem", "DBH", "H",
            "n_cyl", "orders", "length", "seconds"]
    with open(ROOT / "pytls_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("wrote", ROOT / "pytls_metrics.csv")


if __name__ == "__main__":
    main()
