"""Ported code must give what the code it replaced gave (tests/parity)."""

import numpy as np
import pytest
from parity.harness import GOLDEN, cases, flatten, modules

PARAMS = [(m, name) for m in modules() if (GOLDEN / f"{m}.npz").exists() for name in cases(m)]


def missing_goldens(module_names, golden_dir):
    """The case modules with no recording in ``golden_dir``."""
    return [m for m in module_names if not (golden_dir / f"{m}.npz").exists()]


def test_every_case_module_has_a_recording():
    # Without this a deleted or never-captured golden file drops its cases
    # from the parametrisation and the suite stays green.
    missing = missing_goldens(modules(), GOLDEN)
    assert not missing, f"no golden .npz for {missing}; record them with tests/parity/capture.py"


def test_a_missing_recording_is_noticed(tmp_path):
    assert missing_goldens(["canopy"], tmp_path) == ["canopy"]
    assert missing_goldens(["canopy"], GOLDEN) == []


@pytest.mark.parametrize("module,name", PARAMS, ids=[f"{m}:{n}" for m, n in PARAMS])
def test_matches_the_recording(module, name):
    golden = np.load(GOLDEN / f"{module}.npz")
    got = flatten(name, cases(module)[name]())
    expected = {k: golden[k] for k in golden.files if k.startswith(f"{name}/")}
    assert set(got) == set(expected), (sorted(got), sorted(expected))
    for k, want in expected.items():
        have = got[k]
        assert have.shape == want.shape, (k, have.shape, want.shape)
        if want.dtype.kind in "fc":
            np.testing.assert_allclose(have, want, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=k)
        else:
            np.testing.assert_array_equal(have, want, err_msg=k)
