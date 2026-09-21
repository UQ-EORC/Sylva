"""Render a QSM over its point cloud as side-view PNGs (for eyeballing fits).

Usage: python benchmarks/render_qsm.py SITE TREE_ID [OUT.png] [--csv path]
Cylinders are drawn as projected rectangles of their true width; stem in
red, branches in orange, wood points black, other tree points grey.
"""

import csv
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import sylva  # noqa: E402
from sylva import qsm  # noqa: E402
from paths import DATA, OUT


def draw(ax, tree, wood, model, i, j, xlim=None, zlim=None, title=""):
    sel = np.ones(len(tree), bool)
    if xlim:
        sel &= (tree.xyz[:, i] >= xlim[0]) & (tree.xyz[:, i] <= xlim[1])
    if zlim:
        sel &= (tree.xyz[:, j] >= zlim[0]) & (tree.xyz[:, j] <= zlim[1])
    ax.scatter(tree.xyz[sel][::10, i], tree.xyz[sel][::10, j], s=0.3, c="lightgray")
    ws = np.ones(len(wood), bool)
    if xlim:
        ws &= (wood.xyz[:, i] >= xlim[0]) & (wood.xyz[:, i] <= xlim[1])
    if zlim:
        ws &= (wood.xyz[:, j] >= zlim[0]) & (wood.xyz[:, j] <= zlim[1])
    ax.scatter(wood.xyz[ws][:, i], wood.xyz[ws][:, j], s=0.4, c="k")
    s, e, r = model.start, model.end, model.column("radius")
    order = model.column("branch_order")
    for k in range(len(model)):
        d = e[k] - s[k]
        n = np.array([-d[j], d[i]])
        nn = np.linalg.norm(n)
        if nn == 0:
            continue
        n = n / nn * r[k]
        p = np.array([[s[k, i], s[k, j]] + n, [e[k, i], e[k, j]] + n,
                      [e[k, i], e[k, j]] - n, [s[k, i], s[k, j]] - n])
        ax.add_patch(Polygon(p, closed=True, facecolor="red" if order[k] == 0 else "orange",
                             edgecolor="darkred", lw=0.3, alpha=0.5))
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(*xlim)
    if zlim:
        ax.set_ylim(*zlim)
    ax.set_title(title)


def main():
    site, tid = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else f"qsm_{site}_{tid}.png"
    csv_path = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
    root = DATA / site / OUT
    pc = sylva.read(root / "segmented.laz")
    row = next(r for r in csv.DictReader(open(root / "trees.csv")) if int(r["tree_id"]) == tid)
    tree = pc[pc.attrs["tree_id"] == tid]
    wood = qsm.wood_points(tree)
    model = qsm.QSM.from_csv(csv_path or root / "qsm" / f"tree_{tid}.csv")
    bx, by = float(row["x"]), float(row["y"])
    zb = tree.z.min()
    fig, axes = plt.subplots(1, 3, figsize=(21, 10))
    draw(axes[0], tree, wood, model, 0, 2, (bx - 8, bx + 8), (zb - 0.5, zb + 26), f"{site} tree {tid} x-z")
    draw(axes[1], tree, wood, model, 0, 2, (bx - 3, bx + 3), (zb - 0.5, zb + 8), "lower stem x-z")
    draw(axes[2], tree, wood, model, 1, 2, (by - 3, by + 3), (zb - 0.5, zb + 8), "lower stem y-z")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print(out, len(model), "cylinders")


if __name__ == "__main__":
    main()
