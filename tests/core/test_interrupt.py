# Ctrl-C reaches work running in the compiled core: the binding lets Python
# run its signal handlers while it waits, asks the core to stop, and raises
# KeyboardInterrupt once the work has unwound at its next progress count.
import os
import signal
import sys
import threading
import time

import numpy as np
import pytest

from sylva import _core


# The grid the pulses cross on their way out, not only where they end.
BOUNDS = ((0.0, -1.6, -1.6), (30.5, 1.6, 1.6))


def _fan(n: int, seed: int = 0) -> dict:
    """`n` pulses in a narrow fan along +x, one echo each at 30 m."""
    rng = np.random.default_rng(seed)
    d = np.column_stack([np.ones(n), rng.uniform(-0.05, 0.05, n), rng.uniform(-0.05, 0.05, n)])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return {
        "origin": np.zeros((n, 3)),
        "direction": d,
        "echo_start": np.arange(n, dtype=np.int64),
        "echo_count": np.ones(n, dtype=np.int64),
        "echo_range": np.full(n, 30.0),
        "echo_attrs": {},
    }


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill cannot deliver SIGINT to the process itself on Windows")
def test_ctrl_c_stops_a_ray_trace_part_way():
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers only run on the main thread")
    shots = _fan(600_000)

    start = time.perf_counter()
    grid = _core.ray_voxelize(shots, 0.05, BOUNDS)
    full = time.perf_counter() - start
    assert grid.shape[0] > 0
    del grid
    if full < 1.0:
        pytest.skip(f"the trace takes {full:.2f} s here, too short to interrupt reliably")

    timer = threading.Timer(0.2 * full, os.kill, (os.getpid(), signal.SIGINT))
    start = time.perf_counter()
    returned = False
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            _core.ray_voxelize(shots, 0.05, BOUNDS)
            returned = True
            time.sleep(2.0 * full + 5.0)  # the signal, if late, still lands in here
    finally:
        timer.cancel()
    taken = time.perf_counter() - start
    assert not returned, "the trace ran to its end instead of stopping"
    assert taken < 0.75 * full, f"interrupted after {taken:.2f} s of a {full:.2f} s trace"

    # Nothing is left cancelled: the next call runs to its end, and Python's
    # own progress stages still count.
    assert _core.ray_voxelize(_fan(20_000), 0.5, BOUNDS).shape[0] > 0
    task = _core.ProgressTask("after an interrupt", 2)
    task.inc()
    task.set(2)
    task.close()


def test_uninterrupted_work_is_unchanged():
    shots = _fan(20_000, seed=3)
    a = _core.ray_voxelize(shots, 0.5, BOUNDS)
    b = _core.ray_voxelize(shots, 0.5, BOUNDS)
    np.testing.assert_array_equal(a.field("num_hits"), b.field("num_hits"))
    assert a.field("num_hits").sum() > 0
