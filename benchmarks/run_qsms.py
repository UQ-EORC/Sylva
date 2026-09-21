"""Build a QSM for every segmented tree of a site.

Reads ``<site>/pytls/segmented.laz`` and ``trees.csv`` (from
``evaluate_trees.py``), keeps trees with enough points, separates wood by
local anisotropy, thins to 2 cm, builds the QSM and writes ``qsm/tree_<id>.csv``, ``qsm/tree_<id>_trees.txt``
(raycloudtools format) and ``qsm/qsms.csv`` with per-tree totals.

Usage: ``python benchmarks/run_qsms.py SITE [SITE ...] [--min-points 2000]``
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

import sylva
from sylva import qsm
from paths import DATA, OUT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites", nargs="+")
    ap.add_argument("--data", default=str(DATA), help="folder of site folders (default: $SYLVA_DATA or ~/data)")
    ap.add_argument("--min-points", type=int, default=2000)
    ap.add_argument("--min-height", type=float, default=3.0)
    ap.add_argument("--params", default="{}", help="extra build_qsm keyword arguments (Python dict literal)")
    ap.add_argument("--wood", default="{}", help="extra wood_points keyword arguments (Python dict literal)")
    ap.add_argument("--out", default="qsm", help="output directory name under <site>/pytls")
    args = ap.parse_args()
    extra = dict(eval(args.params))
    wood_kw = dict(eval(args.wood))
    for site in args.sites:
        root = Path(args.data) / site / OUT
        out = root / args.out
        out.mkdir(exist_ok=True)
        t0 = time.time()
        pc = sylva.read(root / "segmented.laz")
        labels = pc.attrs["tree_id"]
        rows = list(csv.DictReader(open(root / "trees.csv")))
        summary = []
        plot_meshes = []
        rng = np.random.default_rng(0)
        for r in rows:
            tid = int(r["tree_id"])
            n = int(r["n_points"])
            if n < args.min_points or float(r["height"]) < args.min_height:
                continue
            tree = pc[labels == tid]
            wood = qsm.wood_points(tree, k=20, threshold=0.85, voxel_size=0.02, **wood_kw)
            if len(wood) < 100:
                continue
            dbh = float(r["dbh"])
            t1 = time.time()
            try:
                # No radius cap from the slice DBH: buttressed and fluted stems
                # are wider than their breast-height circle (see the harvest
                # benchmark); the DBH only seeds the allometry fallback.
                model = qsm.build_qsm(wood, base_xy=(float(r["x"]), float(r["y"])),
                                      max_radius=1.5, base_radius=dbh / 2, **extra)
            except ValueError as exc:
                print(f"[{site}] tree {tid}: {exc}")
                continue
            if len(model) == 0:
                continue
            model.to_csv(out / f"tree_{tid}.csv")
            model.to_treefile(out / f"tree_{tid}_trees.txt")
            model.to_obj(out / f"tree_{tid}.obj")
            v, fc, _ = model.mesh(12)
            rgb = rng.integers(40, 230, 3, dtype=np.uint8)
            plot_meshes.append((v, fc, np.tile(rgb, (len(fc), 1))))
            s = model.summary()
            summary.append({
                "tree_id": tid, "x": r["x"], "y": r["y"], "dbh_detected": dbh,
                "height": r["height"], "n_points": n, "n_wood_points": len(wood),
                **s, "seconds": round(time.time() - t1, 2),
            })
        # One mesh for the whole plot, faces coloured per tree.
        offset = 0
        vs, fs, cs = [], [], []
        for v, fc, rgb in plot_meshes:
            vs.append(v); fs.append(fc + offset); cs.append(rgb); offset += len(v)
        qsm.write_ply_mesh(out / "plot_qsm.ply", np.vstack(vs), np.vstack(fs), np.vstack(cs))
        with open(out / "qsms.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
        vol = np.array([s["total_volume_m3"] for s in summary])
        dq = np.array([s["dbh_m"] for s in summary])
        dd = np.array([s["dbh_detected"] for s in summary])
        ok = np.isfinite(dq) & (dd > 0)
        print(f"[{site}] {len(summary)} QSMs in {time.time() - t0:.0f}s; total volume {vol.sum():.1f} m3 "
              f"(median {np.median(vol):.3f}); QSM/detected DBH ratio median "
              f"{np.median(dq[ok] / dd[ok]):.2f}; cylinders median "
              f"{np.median([s['n_cylinders'] for s in summary]):.0f}; -> {out}", flush=True)


if __name__ == "__main__":
    main()
