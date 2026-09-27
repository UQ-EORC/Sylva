"""Write Python outputs for the R package's synthetic-scene tests.

    python tests/parity/export_r_synthetic.py

The R tests (r/sylva/tests/testthat/test-synthetic-python.R) build the
same scenes with the same seeds and require the Python results to 1e-9:
sizes, class counts, column sums and every hundredth point.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import write_expected  # noqa: E402

from parity.export_r_fixtures import OUT  # noqa: E402
from sylva import synthetic  # noqa: E402

TREES = [(2.0, 2.5, 0.2, 1.2), (4.0, 1.0, 0.1, 0.8)]


def cloud_summary(c):
    out = {"n": float(len(c)), "sum": c.xyz.sum(axis=0), "rows": c.xyz[::100]}
    for k, v in c.attrs.items():
        out[k] = np.bincount(np.asarray(v, np.int64)).astype(float)
    return out


def shots_summary(s):
    return {
        "n_shots": float(s.n_shots),
        "echo_count": s.echo_count.astype(float),
        "echo_range": s.echo_range,
        "direction": s.direction[::50],
        "direction_sum": s.direction.sum(axis=0),
        "classification": np.asarray(s.echo_attrs["classification"], float),
        "tree_id": np.asarray(s.echo_attrs["tree_id"], float),
    }


def main():
    d = OUT / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5.0, 25.0, 13)
    y = np.linspace(30.0, -3.0, 13)
    e = {
        "terrain": synthetic.terrain_height(x, y),
        "terrain_slope": synthetic.terrain_height(x, 2.0, slope=0.2),
    }
    t = synthetic.tree(
        1.0, -1.0, dbh=0.25, height=1.4, z0=0.2, n_branches=3, leaf_points=120, seed=6
    )
    e["tree"] = cloud_summary(t)
    e["tree_leaf_area"] = synthetic.leaf_area(t)
    f = synthetic.forest(TREES, size=5.0, ground_points=500, margin=1.0, seed=3)
    e["forest"] = cloud_summary(f)
    e["forest_leaf_area"] = synthetic.leaf_area(f)
    e["scan"] = shots_summary(
        synthetic.scan(
            f, origin=(3.0, 3.0, 1.0), resolution_deg=4.0, max_echoes=3, echo_separation=0.3
        )
    )
    write_expected(d, e)


if __name__ == "__main__":
    main()
