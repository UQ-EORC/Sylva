"""Parity cases: fixed inputs, recorded outputs.

Each ``cases_<module>.py`` defines ``CASES = {name: fn}``; ``fn()`` builds
its inputs from a seeded NumPy generator (never from sylva.synthetic, whose
random streams may change) and returns a flat dict of arrays or scalars.
``capture.py`` records them from the implementation being replaced, and
``tests/test_parity.py`` checks the current one against the recording.
"""

import importlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"


def modules():
    return sorted(p.stem[len("cases_"):] for p in HERE.glob("cases_*.py"))


def cases(module):
    return importlib.import_module(f"parity.cases_{module}").CASES


def flatten(name, out):
    """``{case/key: array}`` from one case's output."""
    flat = {}
    for k, v in out.items():
        flat[f"{name}/{k}"] = np.asarray(v)
    return flat
