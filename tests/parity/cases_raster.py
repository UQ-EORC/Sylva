"""Parity cases for sylva.raster."""

import tempfile
from pathlib import Path

import numpy as np

from sylva import Raster


def grid(seed=7, holes=True):
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 2.0, (11, 14))
    if holes:
        data[rng.uniform(size=data.shape) < 0.2] = np.nan
    return Raster(data, 100.25, -40.5, 0.4)


def geometry():
    r = grid()
    rng = np.random.default_rng(8)
    x = rng.uniform(r.xmin - 1.0, r.xmax + 1.0, 300)
    y = rng.uniform(r.ymin - 1.0, r.ymax + 1.0, 300)
    row, col = r.cell_index(x, y)
    X, Y = r.cell_centers()
    s_row, s_col = r.cell_index(101.0, -39.0)
    return {"row": row, "col": col, "X": X, "Y": Y, "scalar_rc": np.array([s_row, s_col]),
            "extent": np.array([r.xmax, r.ymax, *r.shape])}


def sampling():
    r = grid()
    filled = r.fill_nearest()
    rng = np.random.default_rng(9)
    x = rng.uniform(r.xmin - 1.0, r.xmax + 1.0, (20, 15))
    y = rng.uniform(r.ymin - 1.0, r.ymax + 1.0, (20, 15))
    return {"filled": filled.data, "sample_holes": r.sample(x, y), "sample_filled": filled.sample(x, y),
            "sample_scalar": filled.sample(101.3, -38.2)}


def ascii_grid():
    r = grid()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "g.asc"
        r.to_ascii_grid(p)
        back = Raster.from_ascii_grid(p)
        p2 = Path(d) / "g2.asc"
        r.to_ascii_grid(p2, nodata=-1.0)
        back2 = Raster.from_ascii_grid(p2)
        text = p2.read_text()
    return {"data": back.data, "origin": np.array([back.xmin, back.ymin, back.resolution]),
            "data_nodata": back2.data, "header": np.array(text.splitlines()[:6])}


CASES = {"geometry": geometry, "sampling": sampling, "ascii_grid": ascii_grid}
