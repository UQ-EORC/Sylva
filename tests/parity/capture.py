"""Record the parity outputs of the current implementation.

    python tests/parity/capture.py MODULE [MODULE ...]

Run on the commit *before* a module is ported, and commit the .npz files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity.harness import GOLDEN, cases, flatten  # noqa: E402

for module in sys.argv[1:]:
    flat = {}
    for name, fn in cases(module).items():
        flat.update(flatten(name, fn()))
    GOLDEN.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN / f"{module}.npz", **flat)
    print(f"{module}: {len(flat)} arrays")
