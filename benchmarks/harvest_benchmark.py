"""Score Sylva QSMs against the destructive-harvest reference, beside
the other methods the harvest benchmark holds results for, using that
benchmark's cohort, density assignment and agreement statistics
(``benchmark_ref.py``).

Prints R2 / CCC / bias tables for wood volume, DBH and height on the trees
where all three methods produced a result, and writes
``$SYLVA_HARVEST/results/harvest/per_tree_pytls.csv``.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from paths import HARVEST_CODE

sys.path.insert(0, str(HARVEST_CODE))
import benchmark_ref as B  # noqa: E402

ROOT = B.ROOT
RES = B.RES


def main():
    ref = pd.read_csv(f"{B.NC}/reference/reference.csv")
    ray = B.collect()
    ray["h_cloud"] = ray.cloud.map(B.cloud_extents(ray.cloud.tolist()))
    py = pd.read_csv(f"{ROOT}/pytls_metrics.csv")
    py = py.rename(columns={"V": "pytls_V", "DBH": "pytls_DBH_m", "H": "pytls_H",
                            "n_cyl": "pytls_n_cyl", "status": "pytls_status",
                            "seconds": "pytls_seconds", "dbh_slice_m": "pytls_slice_DBH_m"})
    py["pytls_DBH_cm"] = py.pytls_DBH_m * 100
    py["pytls_slice_DBH_cm"] = py.pytls_slice_DBH_m * 100
    df = ray.merge(py[["cloud", "pytls_status", "pytls_V", "pytls_DBH_cm", "pytls_slice_DBH_cm",
                       "pytls_H", "pytls_n_cyl", "pytls_seconds"]], on="cloud", how="left")
    df = df.merge(ref, on="tree_id", how="left")
    df = B.assign_density(df)
    df["pytls_AGB"] = df.pytls_V * df.rho_kg_m3
    # Every method with results in the benchmark (columns <method>_status), Sylva last.
    others = [c[:-7] for c in ray.columns if c.endswith("_status")]
    label = {"rct": "rayextract"}
    ok = df.pytls_status == "ok"
    for m in others:
        ok &= df[f"{m}_status"] == "ok"
    coh = df[ok & df.volume_ref_m3.notna() & ~df.study.isin(B.EXCLUDED)].copy()
    os.makedirs(RES, exist_ok=True)
    coh.to_csv(f"{RES}/per_tree_pytls.csv", index=False)
    print(f"cohort: {len(coh)} trees with every method and a harvest volume; "
          f"studies {coh.study.value_counts().to_dict()}")
    comparisons = [
        ("Wood volume vs destructive harvest (m3)", "volume_ref_m3",
         [(label.get(m, m), f"{m}_V") for m in others] + [("pytls", "pytls_V")]),
        ("DBH vs harvest tape (cm)", "dbh_ref_cm",
         [(label.get(m, m), f"{m}_DBH_cm") for m in others] + [("pytls QSM", "pytls_DBH_cm"),
          ("pytls slices", "pytls_slice_DBH_cm")]),
        ("Height vs felled tree (m)", "height_ref_m",
         [(label.get(m, m), f"{m}_H") for m in others] + [("pytls", "pytls_H"),
          ("cloud extent", "h_cloud")]),
    ]
    for title, refcol, methods in comparisons:
        cols = [c for _, c in methods if c in coh]
        sub = coh.dropna(subset=[refcol] + cols)
        print(f"\n{title}  (n = {len(sub)})")
        print(f"  {'method':14s} {'bias%':>7s} {'rRMSE%':>7s} {'MAPE%':>7s} {'r2':>6s} {'CCC':>6s} {'slope':>6s}")
        for name, col in methods:
            if col not in cols:
                continue
            a = B.agreement(sub[col], sub[refcol])
            print(f"  {name:14s} {a['rBias_pct']:7.1f} {a['rRMSE_pct']:7.1f} {a['MAPE_pct']:7.1f} "
                  f"{a['r2']:6.3f} {a['CCC']:6.3f} {a['slope0']:6.3f}")
    # Per-study volume bias, the diagnostic that separates cloud/site effects.
    print("\nVolume bias % by study:")
    for study, g in coh.groupby("study"):
        parts = []
        for name, col in [(m, f"{m}_V") for m in others + ["pytls"]]:
            a = B.agreement(g[col], g.volume_ref_m3)
            parts.append(f"{name} {a.get('rBias_pct', float('nan')):+6.1f}")
        print(f"  {study:12s} n={len(g):3d}  " + "  ".join(parts))
    big = coh[coh.dbh_ref_cm >= 40]
    if len(big) >= 3:
        print(f"\nTrees with DBH >= 40 cm (n = {len(big)}): volume bias% "
              + ", ".join(f"{n} {B.agreement(big[c], big.volume_ref_m3)['rBias_pct']:+.1f}"
                          for n, c in [(m, f"{m}_V") for m in others + ["pytls"]]))


if __name__ == "__main__":
    main()
