"""Learn rimta5's keep/drop and zone decision from its own output.

Inputs
  features : per-echo records written by Sylva with ``SYLVA_MTA_FEATURES`` set
             (FEATURE_LEN = 10 + 6*8 floats: line, shot, idx, -, r1, amp, dev,
             late, rule_zone, rule_stage, then per zone range, reflectance,
             votes_own, votes_nb, anchors, density)
  dump     : the rxpmta dump of the same scan (echo order = line, shot, echo);
             see riegl-data/mta/rxpmta.cpp
  truth    : validate.py output for the whole scan (truth.npz, rdbx_side.npz)

One training row per (echo, zone hypothesis k <= KMAX); label 1 when rimta5
kept the echo in zone k. A gradient-boosted classifier scores the rows; per
echo the best-scoring zone is taken and the echo kept when that score exceeds
a threshold tuned on held-out lines. The model is exported as JSON that
Sylva's Rust reader evaluates (crates/sylva-rs/src/io/mta_model.rs).

usage: python train_mta.py <features.bin> <dump.bin> <truth_dir> <out_dir> [--kmax 3]
       [--train-mod 8] [--iters 300]
"""

import argparse
import json
import os
import time

import numpy as np

MAXZ = 8
# record layout of the rxpmta dump: the reference echo order
REC = np.dtype([('time', '<f8'), ('line', '<u4'), ('shot', '<u4'),
                ('tidx', 'u1'), ('tcnt', 'u1'), ('facet', 'u1'), ('flags', 'u1'),
                ('zone_std', '<u2'), ('nzone', '<u2'), ('rng_std', '<f8'),
                ('amp', '<f4'), ('refl', '<f4'), ('dev', '<f4'), ('bg', '<f4'),
                ('rng_zone', '<f4', (MAXZ,)), ('xyz_zone', '<f4', (MAXZ, 3)),
                ('ox', '<f4'), ('oy', '<f4'), ('oz', '<f4'),
                ('dx', '<f4'), ('dy', '<f4'), ('dz', '<f4')])
FEATURE_LEN = 10 + 6 * MAXZ
# feature names in the order the Rust evaluator produces them (see mta_model.rs)
ROW_FEATURES = [
    "zone", "range_k", "refl_k", "votes_own_k", "votes_nb_k", "anchors_k", "density_k",
    "votes_own_other", "votes_nb_other", "anchors_other", "density_other",
    "votes_own_sum", "anchors_sum", "density_sum",
    "amplitude", "deviation", "r1", "late", "n_echoes", "idx_in_shot", "n_zones",
    "prev_amp", "prev_gap", "next_amp", "shot_frac",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(features, dump, truth_dir):
    """Features, and per feature row: kept by rimta5, its zone, echoes in the shot."""
    F = np.memmap(features, dtype=np.float32, mode="r").reshape(-1, FEATURE_LEN)
    d = np.memmap(dump, dtype=REC, mode="r")
    T = np.load(f"{truth_dir}/truth.npz")
    side = np.load(f"{truth_dir}/rdbx_side.npz")
    kept = T["kept"]
    ri = T["rdb_index"]
    tz = np.zeros(len(d), np.uint8)
    tz[kept] = side["best_zone"][ri[kept]]
    log(f"features {len(F):,} rows, dump {len(d):,} echoes, rimta5 kept {kept.sum():,}")

    # join on (line, shot, r1 in 0.1 mm): the feature file may lack pseudo echoes
    def key(line, shot, r1):
        return ((line.astype(np.int64) << 40) | (shot.astype(np.int64) << 20)
                | np.round(r1 * 1e4).astype(np.int64))

    kd = key(d["line"], d["shot"], d["rng_std"].astype(np.float32))
    kf = key(F[:, 0], F[:, 1], F[:, 4])
    od = np.argsort(kd, kind="stable")
    kds = kd[od]
    pos = np.searchsorted(kds, kf)
    ok = pos < len(kds)
    ok[ok] &= kds[pos[ok]] == kf[ok]
    log(f"joined {ok.mean() * 100:.4f}% of feature rows to the dump")
    di = od[np.minimum(pos, len(kds) - 1)]
    return F, kept[di] & ok, tz[di] * ok, d["tcnt"][di]


def shot_context(F):
    """Per echo: strongest amplitude among closer echoes of the same shot (-1 if none)
    and the range gap to the closest closer echo (-1 if none)."""
    n = len(F)
    key = (F[:, 0].astype(np.int64) << 20) | F[:, 1].astype(np.int64)
    r1 = F[:, 4]
    amp = F[:, 5]
    order = np.lexsort((r1, key))
    ks = key[order]
    rs = r1[order]
    am = amp[order]
    idx = np.arange(n)
    pm = np.full(n, -1.0, np.float32)
    gp = np.full(n, -1.0, np.float32)
    for off in range(1, 9):
        j = idx - off
        ok = j >= 0
        ok[ok] &= ks[j[ok]] == ks[ok]
        pm[ok] = np.maximum(pm[ok], am[j[ok]])
        if off == 1:
            gp[ok] = rs[ok] - rs[j[ok]]
    nm = np.full(n, -1.0, np.float32)            # strongest farther echo
    for off in range(1, 9):
        j = idx + off
        ok = j < n
        ok[ok] &= ks[j[ok]] == ks[ok]
        nm[ok] = np.maximum(nm[ok], am[j[ok]])
    prev_amp = np.full(n, -1.0, np.float32)
    prev_gap = np.full(n, -1.0, np.float32)
    next_amp = np.full(n, -1.0, np.float32)
    prev_amp[order] = pm
    prev_gap[order] = gp
    next_amp[order] = nm
    return prev_amp, prev_gap, next_amp


def rows(F, kept, tz, tcnt, kmax):
    """Expand echoes into (echo, zone) rows with the ROW_FEATURES columns."""
    prev_amp, prev_gap, next_amp = shot_context(F)
    # position of the shot within its line (theta proxy: sky above, ground below)
    line = F[:, 0].astype(np.int64)
    shots_per_line = np.zeros(int(line.max()) + 1, np.float32)
    np.maximum.at(shots_per_line, line, F[:, 1] + 1.0)
    shot_frac = F[:, 1] / shots_per_line[line]
    zr = F[:, 10:10 + MAXZ]
    zf = F[:, 10 + MAXZ:10 + 2 * MAXZ]
    vo = F[:, 10 + 2 * MAXZ:10 + 3 * MAXZ]
    vn = F[:, 10 + 3 * MAXZ:10 + 4 * MAXZ]
    an = F[:, 10 + 4 * MAXZ:10 + 5 * MAXZ]
    de = F[:, 10 + 5 * MAXZ:10 + 6 * MAXZ]
    finite = np.isfinite(zr[:, :kmax])
    n_zones = finite.sum(1).astype(np.float32)
    X, y, echo, zone = [], [], [], []
    for k in range(kmax):
        m = finite[:, k]
        if not m.any():
            continue
        others = [j for j in range(kmax) if j != k]

        def other_max(a, others=others):
            return np.where(np.isfinite(zr[:, others]), a[:, others], -1.0).max(1)

        cols = [
            np.full(m.sum(), k + 1, np.float32), zr[m, k], zf[m, k], vo[m, k], vn[m, k],
            an[m, k], de[m, k],
            other_max(vo)[m], other_max(vn)[m], other_max(an)[m], other_max(de)[m],
            np.maximum(vo[:, :kmax], 0).sum(1)[m], np.maximum(an[:, :kmax], 0).sum(1)[m],
            np.maximum(de[:, :kmax], 0).sum(1)[m],
            F[m, 5], F[m, 6], F[m, 4], F[m, 7], tcnt[m].astype(np.float32), F[m, 2], n_zones[m],
            prev_amp[m], prev_gap[m], next_amp[m], shot_frac[m],
        ]
        X.append(np.stack(cols, 1).astype(np.float32))
        y.append(kept[m] & (tz[m] == k + 1))
        echo.append(np.flatnonzero(m))
        zone.append(np.full(m.sum(), k + 1, np.uint8))
    return np.concatenate(X), np.concatenate(y), np.concatenate(echo), np.concatenate(zone)


def decide(p, echo, zone, n, thr):
    """Per echo: best zone by score; keep if score >= thr."""
    order = np.lexsort((-p, echo))
    first = np.concatenate([[True], echo[order][1:] != echo[order][:-1]])
    best = order[first]
    pz = np.zeros(n, np.uint8)
    pk = np.zeros(n, bool)
    pz[echo[best]] = zone[best]
    pk[echo[best]] = p[best] >= thr
    return pz, pk


def score(name, pk, pz, kept, tz):
    inter = np.sum(pk & kept)
    agree = inter / np.sum(pk | kept) * 100
    zacc = np.mean(pz[pk & kept] == tz[pk & kept]) * 100
    print(f"  {name:20s} set agreement {agree:6.3f}%  zone accuracy {zacc:6.3f}%  "
          f"ours {pk.sum():,} rimta5 {kept.sum():,}  wrong-rej {np.sum(kept & ~pk):,}  "
          f"wrong-keep {np.sum(pk & ~kept):,}")
    return agree


def export(model, path, kmax, thr):
    """sklearn HistGradientBoostingClassifier -> plain JSON trees."""
    trees = []
    for stage in model._predictors:
        nodes = stage[0].nodes
        trees.append({
            "value": nodes["value"].astype(float).tolist(),
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": nodes["num_threshold"].astype(float).tolist(),
            "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "leaf": nodes["is_leaf"].astype(int).tolist(),
        })
    doc = {
        "format": "sylva-mta-model/1",
        "features": ROW_FEATURES,
        "kmax": kmax,
        "baseline": float(np.ravel(model._baseline_prediction)[0]),
        "threshold": thr,
        "trees": trees,
    }
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("dump")
    ap.add_argument("truth_dir")
    ap.add_argument("out")
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--train-mod", type=int, default=8,
                    help="validate on lines with line %% mod == mod//2; train on the rest "
                         "(or only on line %% mod == 0 with --small)")
    ap.add_argument("--small", action="store_true", help="train on one eighth of the lines only")
    ap.add_argument("--max-rows", type=float, default=60e6,
                    help="subsample training rows to this many")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="keep threshold written into the model (0 = pick the best agreement)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from sklearn.ensemble import HistGradientBoostingClassifier

    F, kept, tz, tcnt = load(args.features, args.dump, args.truth_dir)
    line = F[:, 0].astype(np.int64)
    amp = F[:, 5]
    r1 = F[:, 4]
    # the deterministic filters stay outside the model
    valid = (amp > 0) & ~((amp < 2.5) & (r1 < 18.0))
    va_res = args.train_mod // 2
    va = (line % args.train_mod == va_res) & valid
    tr = ((line % args.train_mod == 0) if args.small else (line % args.train_mod != va_res)) & valid
    log(f"train echoes {tr.sum():,}  validation echoes {va.sum():,}")

    Xt, yt, _, _ = rows(F[tr], kept[tr], tz[tr], tcnt[tr], args.kmax)
    if len(Xt) > args.max_rows:
        sel = np.random.default_rng(0).choice(len(Xt), int(args.max_rows), replace=False)
        Xt, yt = Xt[sel], yt[sel]
    log(f"train rows {len(Xt):,}, positives {yt.mean() * 100:.2f}%")
    model = HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.1,
                                           max_leaf_nodes=args.leaves, min_samples_leaf=100,
                                           l2_regularization=1.0, early_stopping=False,
                                           random_state=0)
    t0 = time.time()
    model.fit(Xt, yt)
    log(f"fit in {time.time() - t0:.0f} s")

    Xv, _, ev, zv = rows(F[va], kept[va], tz[va], tcnt[va], args.kmax)
    pv = model.predict_proba(Xv)[:, 1]
    nv = va.sum()
    keptv = kept[va]
    tzv = tz[va]
    print("validation lines:")
    score("current rule", F[va, 9] > 0, F[va, 8].astype(np.uint8), keptv, tzv)
    best = (0.0, 0.5)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pz, pk = decide(pv, ev, zv, nv, thr)
        a = score(f"model thr={thr}", pk, pz, keptv, tzv)
        if a > best[0]:
            best = (a, thr)
    thr = args.threshold if args.threshold > 0 else best[1]
    log(f"threshold written into the model: {thr}")
    size = export(model, os.path.join(args.out, "mta_model.json"), args.kmax, thr)
    log(f"exported {size / 1e6:.2f} MB model with {len(model._predictors)} trees "
        f"to {args.out}/mta_model.json")


if __name__ == "__main__":
    main()
