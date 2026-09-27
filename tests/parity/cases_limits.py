"""Parity cases for sylva.limits."""

import numpy as np

from sylva import limits


def human():
    values = [0, 1, 999, 999.4, 999.6, 1000, 1049, 1050, 1e6, 12345678, 2.5e9, 7.77e12, 3.2e15, 0.4, -5.0]
    return {"human": np.array([limits.human(v) for v in values])}


def budget():
    try:
        limits.set_budget(2.5)
        b = limits.budget()
        try:
            limits.check(10**6, 5000, "a 100 x 100 x 100 grid", "a larger voxel")
            msg = ""
        except ValueError as e:
            msg = str(e)
        limits.check(1000, 8, "a small thing", "nothing")
    finally:
        limits.set_budget(None)
    return {"budget": b, "message": np.array(msg)}


CASES = {"human": human, "budget": budget}
