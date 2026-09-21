"""Structural comparison of Sylva and rayextract QSMs on the harvest trees.

Every model is reduced to the same cylinder representation (start, end,
radius, branch order, stem flag), then per tree: stem radius at fixed
heights above the base, stem / branch volume, total length, cylinder count,
max branch order. Cross-method ratios (Sylva vs rayextract, and vs any model
given with ``--other``) are summarised per height bin and per study, alongside the harvest volume.

Usage: python benchmarks/harvest_structure.py [--csv out.csv] [--other NAME=DIR]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from paths import HARVEST_CODE, HARVEST_DATA, HARVEST_RESULTS

sys.path.insert(0, str(HARVEST_CODE))
from collect import parse_treeinfo  # noqa: E402

ROOT = str(HARVEST_DATA)
HEIGHTS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0])


# ---------------------------------------------------------------- loaders
def load_libqsm(path):
    """libqsm binary ``.qsm``, format 2: float32 nodes, 26-byte edges."""
    data = open(path, "rb").read()
    end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    hdr = dict(line.split(" ", 1) for line in data[:end].decode().splitlines()
               if " " in line and not line.startswith("#"))
    n_nodes, n_edges = int(hdr["NODES"]), int(hdr["EDGES"])
    off = np.array([float(hdr["X_OFFSET"]), float(hdr["Y_OFFSET"]), float(hdr["Z_OFFSET"])])
    nodes = np.frombuffer(data, dtype="<f4", count=3 * n_nodes, offset=end).reshape(-1, 3).astype(float) + off
    edge_dt = np.dtype([("src", "<u4"), ("tgt", "<u4"), ("r", "<f4"), ("q", "u1"), ("axis", "<u4"),
                        ("order", "u1"), ("sub", "<f4"), ("d2r", "<f4")])
    edges = np.frombuffer(data, dtype=edge_dt, count=n_edges, offset=end + 12 * n_nodes)
    src, tgt = edges["src"].astype(int), edges["tgt"].astype(int)
    if max(src.max(), tgt.max()) >= n_nodes:  # 1-based node ids
        src, tgt = src - 1, tgt - 1
    start = nodes[src]
    endp = nodes[tgt]
    order = edges["order"].astype(int) - 1  # the format numbers the trunk 1
    return dict(start=start, end=endp, radius=edges["r"].astype(float), order=order,
                stem=(edges["axis"] == 1))


def load_rct(path):
    trees = parse_treeinfo(path)
    if not trees:
        return None
    t = max(trees, key=lambda t: t["_segs"].iloc[0]["volume"])
    seg = t["_segs"]
    xyz = seg[["x", "y", "z"]].to_numpy(float)
    r = seg["radius"].to_numpy(float)
    par = seg["parent_id"].to_numpy(int)
    # Branch order: the largest-radius child continues its parent's branch.
    children = {}
    for i, p in enumerate(par):
        if p >= 0:
            children.setdefault(p, []).append(i)
    order = np.zeros(len(seg), int)
    stem = np.zeros(len(seg), bool)
    stack = [i for i, p in enumerate(par) if p < 0]
    stem[stack] = True
    while stack:
        s = stack.pop()
        kids = sorted(children.get(s, []), key=lambda c: -r[c])
        for k, c in enumerate(kids):
            order[c] = order[s] + (0 if k == 0 else 1)
            stem[c] = stem[s] and k == 0
            stack.append(c)
    keep = par >= 0
    return dict(start=xyz[par[keep]], end=xyz[keep], radius=r[keep], order=order[keep], stem=stem[keep])


def load_pytls(path):
    c = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    start = c[:, 0:3]
    endp = start + c[:, 3:6] * c[:, 6:7]
    order = c[:, 9].astype(int)
    return dict(start=start, end=endp, radius=c[:, 7], order=order, stem=order == 0)


# ---------------------------------------------------------------- metrics
def summarise(m):
    L = np.linalg.norm(m["end"] - m["start"], axis=1)
    vol = np.pi * m["radius"] ** 2 * L
    zb = min(m["start"][:, 2].min(), m["end"][:, 2].min())
    zc = (m["start"][:, 2] + m["end"][:, 2]) / 2 - zb
    out = dict(V=vol.sum(), V_stem=vol[m["stem"]].sum(), V_branch=vol[~m["stem"]].sum(),
               L_total=L.sum(), L_stem=L[m["stem"]].sum(), n_cyl=len(L), max_order=int(m["order"].max()),
               n_branches=int(((~m["stem"]) & (m["order"] >= 1)).sum()), H=float(m["end"][:, 2].max() - zb))
    for h in HEIGHTS:
        s = m["stem"]
        if s.any():
            k = np.argmin(np.abs(zc[s] - h))
            out[f"r{h:g}"] = m["radius"][s][k] if abs(zc[s][k] - h) < 1.0 else np.nan
        else:
            out[f"r{h:g}"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(HARVEST_RESULTS / "structure_pytls.csv"))
    ap.add_argument("--other", metavar="NAME=DIR", help="a third set of models to compare with: "
                    "libqsm files at <harvest data>/DIR/<cloud>/<cloud>.qsm")
    args = ap.parse_args()
    other = args.other.split("=", 1) if args.other else None
    names = ["pytls", "rct"] + ([other[0]] if other else [])
    against = names[1:]
    per = pd.read_csv(HARVEST_RESULTS / "per_tree_pytls.csv")
    rows = []
    for _, t in per.iterrows():
        c = t.cloud
        try:
            models = {"pytls": load_pytls(f"{ROOT}/out_pytls_ref/{c}.csv"),
                      "rct": load_rct(f"{ROOT}/out_rct_ref/{c}/{c}_raycloud_trees_info.txt")}
            if other:
                models[other[0]] = load_libqsm(f"{ROOT}/{other[1]}/{c}/{c}.qsm")
        except (OSError, ValueError) as exc:
            print(f"[{c}] skipped: {exc}")
            continue
        row = dict(cloud=c, study=t.study, dbh_ref_cm=t.dbh_ref_cm, volume_ref_m3=t.volume_ref_m3,
                   height_ref_m=t.height_ref_m)
        for name, m in models.items():
            if m is None:
                continue
            for k, v in summarise(m).items():
                row[f"{name}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    print(f"{len(df)} trees -> {args.csv}\n")

    def med(a, b):
        r = df[a] / df[b]
        r = r[np.isfinite(r) & (r > 0)]
        return f"{r.median():.2f} [{r.quantile(0.25):.2f}-{r.quantile(0.75):.2f}] n={len(r)}"

    def ratios(key):
        return "  ".join(f"vs {n} {med('pytls_' + key, n + '_' + key):26s}" for n in against)

    print("Ratios (median [IQR]) of Sylva to " + ", ".join(against) + ":")
    for k in ("V", "V_stem", "V_branch", "L_total", "L_stem", "n_cyl", "n_branches", "H"):
        print(f"  {k:12s} {ratios(k)}")
    print("\nStem radius by height above base, ratio of Sylva to each:")
    for h in HEIGHTS:
        print(f"  {h:4.0f} m  {ratios(f'r{h:g}')}")
    print("\nVolume / harvest by study (median): stem share and totals")
    for s, g in df.groupby("study"):
        parts = []
        for name in names:
            v = g[f"{name}_V"] / g.volume_ref_m3
            share = g[f"{name}_V_stem"] / g[f"{name}_V"]
            parts.append(f"{name}: V/ref {v.median():.2f} stem share {share.median():.2f} branches/ref {(g[f'{name}_V_branch'] / g.volume_ref_m3).median():.2f}")
        print(f"  {s:10s} n={len(g):2d}  " + " | ".join(parts))
    print("\nMax branch order median: " + ", ".join(f"{n} {df[f'{n}_max_order'].median():.0f}" for n in names))


if __name__ == "__main__":
    main()
