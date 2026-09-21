"""Score a directory of per-tree PLY files (e.g. raycloudtools `trees/`) with the same metric."""
import glob
import sys
from pathlib import Path

import numpy as np
from paths import DATA

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_trees import evaluate

import sylva
from sylva.trees import Tree


def load_dir(d):
    xyz, lab, stems = [], [], []
    for i, f in enumerate(sorted(glob.glob(str(d / "tree_*.ply")))):
        pc = sylva.read(f)
        xyz.append(pc.xyz); lab.append(np.full(len(pc), i + 1, np.int64))
        base = pc.xyz[np.argmin(pc.z)]
        stems.append(Tree(i + 1, base[0], base[1], n_points=len(pc)))
    return sylva.PointCloud(np.vstack(xyz)), np.concatenate(lab), stems

for site in sys.argv[1:]:
    root = DATA / site
    ref = np.load(root / "reference" / f"{site}_reference_no_ground.npy")
    for variant in ("raycloudtools_defaults", "raycloudtools_reference_treefiltered"):
        d = root / variant / "trees"
        if not d.is_dir(): continue
        cloud, labels, stems = load_dir(d)
        m = evaluate(cloud, stems, labels, ref, 0.5)
        print(f"[{site}] {variant:38s} trees {len(stems):3d} (outside {m['n_outside_reference']})  P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} IoU {m['mean_iou_detected']:.2f} acc {m['point_accuracy']:.2f}  TP {m['tp']} FP {m['fp']}", flush=True)
