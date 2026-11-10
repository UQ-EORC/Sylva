"""Import the installed wheel and run a little of everything it links.

Run from outside the repository, so that `sylva` is the installed package.
It reads and writes LAS, LAZ and a shots file (the compiled readers and
their compression), and calls into the core with NumPy arrays.
"""
import pathlib
import platform
import sys
import tempfile

import numpy as np

import sylva
from sylva import _core

print(f"sylva {sylva.__version__} (core {_core.__version__}), Python {platform.python_version()}, "
      f"NumPy {np.__version__}, {platform.system()} {platform.machine()}")
assert pathlib.Path(sylva.__file__).resolve().parent.parent.name != "python", "imported the source tree, not the wheel"

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 10.0, (5000, 3))
cloud = sylva.PointCloud(xyz, {"intensity": rng.uniform(0, 1, len(xyz)).astype(np.float32)})

with tempfile.TemporaryDirectory() as tmp:
    for name in ("smoke.las", "smoke.laz"):
        path = pathlib.Path(tmp) / name
        sylva.io.write(cloud, path)
        back = sylva.io.read(path)
        assert len(back) == len(cloud), name
        np.testing.assert_allclose(back.xyz, xyz, atol=2e-3)
    shots = _core.shots_from_pointcloud(xyz, None, (5.0, 5.0, 20.0))
    path = pathlib.Path(tmp) / "smoke.parquet"
    _core.write_shots(shots, str(path))
    assert len(_core.read_shots(str(path))["echo_range"]) == len(shots["echo_range"])

kept = _core.voxel_downsample_indices(xyz, 1.0)
assert 0 < len(kept) <= 1000
distances, indices = _core.knn(xyz, xyz[:10], 3)
assert distances.shape == (10, 3) and np.array_equal(indices[:, 0], np.arange(10))
assert _core.memory_available() is None or _core.memory_available() > 0
print("wheel smoke test passed", file=sys.stderr)
