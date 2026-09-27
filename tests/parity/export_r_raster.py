"""Inputs and Python outputs for r/sylva/tests/testthat/test-raster-python.R.

    python tests/parity/export_r_raster.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import OUT, write_expected  # noqa: E402

from parity import cases_raster as cr  # noqa: E402


def main():
    d = OUT / "raster"
    d.mkdir(parents=True, exist_ok=True)
    r = cr.grid()
    np.savetxt(d / "grid.csv", r.data, delimiter=",", fmt="%.17g")
    rng = np.random.default_rng(8)
    x = rng.uniform(r.xmin - 1.0, r.xmax + 1.0, 300)
    y = rng.uniform(r.ymin - 1.0, r.ymax + 1.0, 300)
    rng = np.random.default_rng(9)
    sx = rng.uniform(r.xmin - 1.0, r.xmax + 1.0, (20, 15))
    sy = rng.uniform(r.ymin - 1.0, r.ymax + 1.0, (20, 15))
    e = {"origin": np.array([r.xmin, r.ymin, r.resolution]), "x": x, "y": y, "sx": sx.ravel(), "sy": sy.ravel()}
    g = cr.geometry()
    e.update({"row": g["row"] + 1, "col": g["col"] + 1, "X": g["X"], "Y": g["Y"],
              "scalar_rc": g["scalar_rc"] + 1, "extent": g["extent"]})
    s = cr.sampling()
    e.update(s)
    a = cr.ascii_grid()
    e.update({"ascii_data": a["data"], "ascii_origin": a["origin"], "ascii_data_nodata": a["data_nodata"],
              "ascii_header": a["header"]})
    write_expected(d, e)


if __name__ == "__main__":
    main()
