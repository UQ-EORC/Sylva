import numpy as np
import pytest

from sylva import PointCloud


def make_terrain(rng, n=20000, size=20.0, slope=0.05, noise=0.01):
    xy = rng.uniform(0, size, (n, 2))
    z = slope * xy[:, 0] + 0.2 * np.sin(xy[:, 1] / 3) + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


def make_stem(rng, cx, cy, radius, height, z0=0.0, density=4000, noise=0.003):
    n = int(density * height)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, height, n)
    r = radius + rng.normal(0, noise, n)
    return np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta), z0 + z])


def make_crown(rng, cx, cy, zc, radius, n=3000):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.6, 1.0, (n, 1)) ** (1 / 3) * radius
    return v + [cx, cy, zc]


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tree_specs():
    # (x, y, dbh, height)
    return [(5.0, 5.0, 0.30, 12.0), (14.0, 6.0, 0.20, 9.0), (8.0, 15.0, 0.45, 15.0)]


@pytest.fixture(scope="session")
def forest(rng, tree_specs):
    """Synthetic sloped terrain with three cylindrical stems and ball crowns.

    Z is *not* normalised: stems start at the local terrain height.
    """
    parts = [make_terrain(rng)]
    for x, y, dbh, h in tree_specs:
        z0 = 0.05 * x + 0.2 * np.sin(y / 3)
        parts.append(make_stem(rng, x, y, dbh / 2, h, z0=z0))
        parts.append(make_crown(rng, x, y, z0 + h - 2.5, 2.5))
    xyz = np.vstack(parts)
    return PointCloud(xyz, {"intensity": rng.integers(0, 65535, len(xyz), dtype=np.uint16)})


@pytest.fixture(scope="session")
def single_tree(rng):
    """One stem (r=0.15, 6 m) with a side branch, already normalised."""
    stem = make_stem(rng, 0, 0, 0.15, 6.0, density=3000)
    # Branch leaving at 4 m, going out along +x and slightly up.
    n = 1500
    t = rng.uniform(0, 2.0, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = 0.05 + rng.normal(0, 0.002, n)
    branch = np.column_stack([
        0.15 + t, r * np.cos(theta), 4.0 + 0.3 * t + r * np.sin(theta)
    ])
    return PointCloud(np.vstack([stem, branch]))
