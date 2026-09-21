"""Evaluate the Sylva tree pipeline against manually segmented reference plots.

Each site directory holds ``input/<site>_raycloud.ply`` and
``reference/<site>_reference_no_ground.npy`` (``(N, 4)``: x, y, z, tree
label; 0 = unassigned). Sylva labels are transferred to the reference points
by nearest neighbour and scored as instance segmentation:

* a reference tree is *detected* when its best-overlapping Sylva tree has
  IoU >= ``--iou`` (default 0.5);
* precision / recall / F1 over trees, mean IoU of detected trees, and the
  point-level fraction of reference tree points carrying the right label.

Usage: ``python benchmarks/evaluate_trees.py SITE [SITE ...] [--data DIR]``
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import sylva
from sylva import filters, ground, trees
from paths import DATA, OUT


def run_pipeline(site_dir: Path, out: Path, log, max_points: int = 40_000_000,
                 thin: float = 0.02) -> tuple[sylva.PointCloud, list, np.ndarray]:
    site = site_dir.name
    t0 = time.time()
    pc = sylva.read(site_dir / "input" / f"{site}_raycloud.ply")
    pc = pc.without("nx", "ny", "nz", "time", "red", "green", "blue", "alpha")
    log(f"read {len(pc):,} points in {time.time() - t0:.1f}s")
    if len(pc) > max_points:
        # Very large plots: voxel-thin so the 5 cm segmentation graph and the
        # per-call Rust copies stay within memory (2 cm, or 5 cm above 60M).
        thin = max(thin, 0.05) if len(pc) > 60_000_000 else thin
        pc = filters.voxel_downsample(pc, thin)
        log(f"thinned to {len(pc):,} points at {thin} m")
    thin = filters.voxel_downsample(pc, 0.05)
    cls = ground.classify_ground_pmf(thin, cell_size=0.5, slope=0.5, max_window=8)
    dtm = ground.make_dtm(cls, 0.5)
    norm = ground.normalize_height(pc, dtm)
    dtm.to_ascii_grid(out / "dtm.asc")
    ground.make_chm(norm, 0.5).to_ascii_grid(out / "chm.asc")
    log(f"ground + DTM done {time.time() - t0:.1f}s")
    cands = trees.detect_stems(norm)
    log(f"{len(cands)} candidates {time.time() - t0:.1f}s")
    # Segmentation graph voxel: 5 cm normally, 10 cm on very large plots.
    graph_voxel = 0.1 if len(norm) > max_points else 0.05
    labels = trees.segment_trees(norm, cands, voxel_size=graph_voxel)
    log(f"segmented (graph voxel {graph_voxel}) {time.time() - t0:.1f}s")
    trees.tree_heights(norm, labels, cands)
    stems, labels = trees.prune_trees(cands, labels)
    crowns = trees.crown_metrics_all(norm, labels)
    log(f"{len(cands)} candidates -> {len(stems)} trees; pipeline {time.time() - t0:.1f}s")
    rows = [{**t.as_dict(), **crowns.get(t.tree_id, {})} for t in stems]
    keys = list(stems[0].as_dict()) + ["crown_area", "crown_base_height", "crown_depth",
                                       "crown_diameter"]
    with open(out / "trees.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    sylva.write(norm.with_attrs(tree_id=labels.astype(np.int32)), out / "segmented.laz")
    return norm, stems, labels


def evaluate(norm: sylva.PointCloud, stems: list, labels: np.ndarray, ref: np.ndarray,
             iou_threshold: float, max_dist: float = 0.1) -> dict:
    ref_xyz = np.ascontiguousarray(ref[:, :3], dtype=np.float64)
    ref_lab = ref[:, 3].astype(np.int64)
    # Transfer labels through a 5 cm thinning of the cloud: same answer at the
    # 0.1 m matching distance, a third of the memory on 100M-point plots.
    keep = filters._core.voxel_downsample_indices(norm.xyz, 0.05)
    d, idx = filters.knn(norm.xyz[keep], ref_xyz, 1)
    idx = keep[idx[:, 0]]
    pred = np.where(d[:, 0] <= max_dist, labels[idx], -1)
    ref_ids = np.unique(ref_lab[ref_lab > 0])
    # Predicted trees with (almost) no points inside the reference's coverage
    # lie outside the reference plot and are not counted as false positives.
    covered = np.zeros(int(labels.max()) + 2, dtype=np.int64)
    total = np.zeros_like(covered)
    np.add.at(total, np.clip(labels, -1, None) + 1, 1)
    np.add.at(covered, np.clip(labels[idx[d[:, 0] <= max_dist]], -1, None) + 1, 1)
    inside = {t.tree_id for t in stems if covered[t.tree_id + 1] >= 0.2 * max(total[t.tree_id + 1], 1)}
    n_outside = len(stems) - len(inside)
    pred_ids = np.array([t.tree_id for t in stems if t.tree_id in inside])
    # Confusion counts between reference trees and predicted trees.
    both = (ref_lab > 0) & (pred > 0)
    pair = ref_lab[both] * (pred.max() + 1) + pred[both]
    u, c = np.unique(pair, return_counts=True)
    r_of = u // (pred.max() + 1)
    p_of = u % (pred.max() + 1)
    ref_size = {int(r): int(n) for r, n in zip(*np.unique(ref_lab[ref_lab > 0], return_counts=True))}
    pred_size = {int(p): int(n) for p, n in zip(*np.unique(pred[pred > 0], return_counts=True))}
    best: dict[int, tuple[int, float]] = {}
    for r, p, n in zip(r_of, p_of, c):
        iou = n / (ref_size[int(r)] + pred_size.get(int(p), 0) - n)
        if int(r) not in best or iou > best[int(r)][1]:
            best[int(r)] = (int(p), float(iou))
    # One-to-one: a predicted tree can only detect one reference tree.
    claimed: dict[int, int] = {}
    tp = []
    for r in sorted(best, key=lambda r: -best[r][1]):
        p, iou = best[r]
        if iou >= iou_threshold and p not in claimed:
            claimed[p] = r
            tp.append((r, p, iou))
    n_tp = len(tp)
    n_fn = len(ref_ids) - n_tp
    n_fp = len(pred_ids) - n_tp
    precision = n_tp / max(len(pred_ids), 1)
    recall = n_tp / max(len(ref_ids), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    # Point-level accuracy on reference tree points: right tree via the TP mapping.
    remap = np.full(pred.max() + 2, -1, dtype=np.int64)
    for r, p, _ in tp:
        remap[p] = r
    correct = (remap[np.clip(pred, -1, pred.max() + 1)] == ref_lab) & (ref_lab > 0)
    point_acc = correct.sum() / max((ref_lab > 0).sum(), 1)
    # Tree top height: compare max z of matched trees.
    top_err = []
    for r, p, _ in tp:
        top_err.append(norm.z[labels == p].max() - ref_xyz[ref_lab == r, 2].max())
    top_err = np.array(top_err)
    return {
        "n_reference": int(len(ref_ids)),
        "n_predicted": int(len(pred_ids)),
        "n_outside_reference": int(n_outside),
        "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "mean_iou_detected": round(float(np.mean([t[2] for t in tp])) if tp else float("nan"), 3),
        "mean_iou_all_reference": round(float(np.mean([best.get(int(r), (0, 0.0))[1] for r in ref_ids])), 3),
        "point_accuracy": round(float(point_acc), 3),
        "ref_points_unmatched_to_cloud": round(float((d[:, 0] > max_dist).mean()), 3),
        "top_height_bias_m": round(float(top_err.mean()) if len(top_err) else float("nan"), 2),
        "top_height_rmse_m": round(float(np.sqrt((top_err**2).mean())) if len(top_err) else float("nan"), 2),
        "iou_threshold": iou_threshold,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sites", nargs="+")
    ap.add_argument("--data", default=str(DATA), help="folder of site folders (default: $SYLVA_DATA or ~/data)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    summary = []
    for site in args.sites:
        site_dir = Path(args.data) / site
        out = site_dir / OUT
        out.mkdir(exist_ok=True)
        log = lambda m, s=site: print(f"[{s}] {m}", flush=True)  # noqa: E731
        norm, stems, labels = run_pipeline(site_dir, out, log)
        ref = np.load(site_dir / "reference" / f"{site}_reference_no_ground.npy", mmap_mode="r")
        stride = max(1, len(ref) // 40_000_000)  # score on <= 40M reference points
        ref = np.ascontiguousarray(ref[::stride])
        if stride > 1:
            log(f"reference subsampled 1/{stride} to {len(ref):,} points")
        metrics = {"site": site, **evaluate(norm, stems, labels, ref, args.iou)}
        (out / "evaluation.json").write_text(json.dumps(metrics, indent=2))
        log(f"P {metrics['precision']:.2f} R {metrics['recall']:.2f} F1 {metrics['f1']:.2f} "
            f"IoU {metrics['mean_iou_detected']:.2f} point-acc {metrics['point_accuracy']:.2f} "
            f"({metrics['tp']}/{metrics['n_reference']} ref trees, {metrics['fp']} FP)")
        summary.append(metrics)
        del norm, labels, ref
    if len(summary) > 1:
        print("\nsite,n_ref,n_pred,tp,fp,fn,precision,recall,f1,mean_iou,point_acc,top_rmse")
        for m in summary:
            print(",".join(str(m[k]) for k in ("site", "n_reference", "n_predicted", "tp", "fp", "fn",
                                              "precision", "recall", "f1", "mean_iou_detected",
                                              "point_accuracy", "top_height_rmse_m")))


if __name__ == "__main__":
    main()
