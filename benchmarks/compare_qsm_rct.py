"""Compare Sylva QSM totals with raycloudtools `_trees.txt` per matched tree."""
import csv
import sys

import os
import numpy as np
from paths import DATA, OUT

QSM_DIR = os.environ.get("QSM_DIR", "qsm")


def rct_trees(path):
    out = []
    for line in open(path):
        if line.startswith(("#", "x,y")): continue
        seg = np.array([list(map(float, s.split(",")[:6])) for s in line.strip().split(", ") if s])
        vol = 0.0
        for k in range(1, len(seg)):
            par = int(seg[k, 4]); vol += np.pi * seg[k, 3]**2 * np.linalg.norm(seg[k, :3] - seg[par, :3])
        z0 = seg[0, 2]; k = np.argmin(np.abs(seg[:, 2] - (z0 + 1.3)))
        out.append((seg[0, 0], seg[0, 1], vol, 2 * seg[k, 3], seg[:, 2].max() - z0))
    return np.array(out)

for site in sys.argv[1:]:
    root = DATA / site
    rct = rct_trees(root / "raycloudtools_defaults" / f"{site}_raycloud_trees.txt")
    rows = list(csv.DictReader(open(root / OUT / QSM_DIR / "qsms.csv")))
    pv, rv, pd, rd = [], [], [], []
    for r in rows:
        d = np.hypot(rct[:, 0] - float(r["x"]), rct[:, 1] - float(r["y"])); j = d.argmin()
        if d[j] < 0.5:
            pv.append(float(r["total_volume_m3"])); rv.append(rct[j, 2]); pd.append(float(r["dbh_m"])); rd.append(rct[j, 3])
    pv, rv, pd, rd = map(np.array, (pv, rv, pd, rd))
    ok = (rv > 0) & (pv > 0)
    ratio = pv[ok] / rv[ok]
    big = ok & (rd > 0.2)
    print(f"[{site}] matched {ok.sum()}/{len(rows)} trees; volume sylva/rct: median {np.median(ratio):.2f} IQR {np.percentile(ratio,25):.2f}-{np.percentile(ratio,75):.2f}; "
          f"totals {pv[ok].sum():.1f} vs {rv[ok].sum():.1f} m3; DBH ratio median {np.median(pd[ok]/rd[ok]):.2f}; "
          f"trees DBH>0.2: n={big.sum()} vol ratio median {np.median(pv[big]/rv[big]) if big.any() else float('nan'):.2f}")
