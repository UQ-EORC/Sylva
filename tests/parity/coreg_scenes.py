"""Seeded forest plots and virtual scans for the coregistration parity cases.

A stand of leaning-free cylinders on undulating terrain, with crowns, low
understorey and plan-view occlusion, scanned from given positions. Built from
NumPy's generator only, so the scenes stay fixed while the package's own
simulators change.
"""

from dataclasses import dataclass, field

import numpy as np

from sylva.coreg import transforms as tf


@dataclass
class Stand:
    trees: np.ndarray  # (n, 5): x, y, base z, dbh, crown base
    coeffs: np.ndarray
    size: float

    def terrain(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        a = self.coeffs
        return a[0] * xy[:, 0] + a[1] * xy[:, 1] + a[2] * np.sin(xy[:, 0] / 9.0 + a[3]) \
            + a[4] * np.cos(xy[:, 1] / 7.0 + a[5])


@dataclass
class Scene:
    stand: Stand
    clouds: list = field(default_factory=list)  # (n, 3) arrays in each scanner's frame
    truth: list = field(default_factory=list)  # world_from_scan
    targets: list = field(default_factory=list)  # (m, 3) target positions in each scan's frame
    positions: np.ndarray = None


def stand(seed, size=44.0, n_trees=48, spacing=2.6, lattice=None):
    """Trees by dart throwing (or on a jittered lattice of the given spacing)."""
    rng = np.random.default_rng(seed)
    coeffs = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), rng.uniform(0, 0.8),
                       rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.8), rng.uniform(0, 2 * np.pi)])
    if lattice:
        g = np.arange(-size / 2 + lattice / 2, size / 2, lattice)
        xy = np.array([[x, y] for x in g for y in g]) + rng.normal(0, 0.03, (len(g) ** 2, 2))
    else:
        pts = []
        while len(pts) < n_trees:
            p = rng.uniform(-size / 2, size / 2, 2)
            if all(np.hypot(*(p - q)) > spacing for q in pts):
                pts.append(p)
        xy = np.array(pts)
    s = Stand(np.zeros((len(xy), 5)), coeffs, size)
    dbh = np.full(len(xy), 0.3) if lattice else np.clip(rng.lognormal(np.log(0.28), 0.4, len(xy)), 0.1, 0.8)
    s.trees = np.column_stack([xy, s.terrain(xy), dbh, rng.uniform(5.0, 8.0, len(xy))])
    return s


def _shadowed(points, scanner, trees, skip):
    d = points[:, :2] - scanner[:2]
    length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    u = d / length[:, None]
    hidden = np.zeros(len(points), bool)
    for k, (bx, by, _, dbh, _) in enumerate(trees):
        if k == skip:
            continue
        to = np.array([bx, by]) - scanner[:2]
        along = u @ to
        perp = np.abs(u[:, 0] * to[1] - u[:, 1] * to[0])
        hidden |= (perp < 0.5 * dbh) & (along > 0.05) & (along < length - 0.5 * dbh)
    return hidden


def scan(s, scanner_xy, rng, max_range=24.0, ground=45000, under=5000, stem_density=22000.0,
         crown=900):
    """World points one scanner sees, and its position."""
    scanner = np.array([*scanner_xy, s.terrain(np.array(scanner_xy))[0] + 1.6])
    parts = []
    for k, (x, y, z0, dbh, top) in enumerate(s.trees):
        dist = np.hypot(x - scanner[0], y - scanner[1])
        if dist > max_range or dist < 0.5:
            continue
        n = int(np.clip(stem_density / dist, 400, 9000))
        h = rng.uniform(0.0, top, n)
        bearing = np.arctan2(scanner[1] - y, scanner[0] - x)
        a = bearing + rng.uniform(-np.pi / 2, np.pi / 2, n)
        r = 0.5 * dbh
        stem = np.column_stack([x + r * np.cos(a), y + r * np.sin(a), z0 + h])
        stem = stem[~_shadowed(stem, scanner, s.trees, k)]
        c = rng.normal(0, 1, (crown, 3)) * [1.8, 1.8, 1.2] + [x, y, z0 + top + 2.5]
        parts += [stem, c]
    rad = rng.uniform(0.6, max_range, ground)
    phi = rng.uniform(0, 2 * np.pi, ground)
    xy = scanner[:2] + np.column_stack([rad * np.cos(phi), rad * np.sin(phi)])
    g = np.column_stack([xy, s.terrain(xy) + rng.normal(0, 0.01, ground)])
    parts.append(g[~_shadowed(g, scanner, s.trees, -1)])
    rad = rng.uniform(0.6, max_range, under)
    phi = rng.uniform(0, 2 * np.pi, under)
    xy = scanner[:2] + np.column_stack([rad * np.cos(phi), rad * np.sin(phi)])
    parts.append(np.column_stack([xy, s.terrain(xy) + rng.exponential(0.35, under)]))
    world = np.vstack(parts)
    world = world[np.linalg.norm(world[:, :2] - scanner[:2], axis=1) <= max_range]
    return world + rng.normal(0, 0.004, world.shape), scanner


def survey(seed, positions, stand_seed=None, yaws=None, targets=None, target_range=30.0, **kw):
    """Scans of a stand, each in its own frame (yawed, scanner at its origin).

    ``targets``: world positions of reflective targets; each scan records the
    ones within ``target_range`` in its own frame.
    """
    s = kw.pop("the_stand", None) or stand(seed if stand_seed is None else stand_seed)
    rng = np.random.default_rng(seed + 1000)
    scene = Scene(s, positions=np.asarray(positions, float))
    for k, p in enumerate(positions):
        world, scanner = scan(s, p, rng, **kw)
        yaw = rng.uniform(-np.pi, np.pi) if yaws is None else yaws[k]
        T = tf.yaw_transform(yaw, *scanner)
        scene.truth.append(T)
        scene.clouds.append(tf.transform_points(tf.invert(T), world))
        if targets is not None:
            t = np.asarray(targets, float)
            seen = t[np.linalg.norm(t[:, :2] - scanner[:2], axis=1) <= target_range]
            scene.targets.append(tf.transform_points(tf.invert(T), seen) + rng.normal(0, 0.002, seen.shape))
    return scene
