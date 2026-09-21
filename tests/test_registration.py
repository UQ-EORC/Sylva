import numpy as np
import pytest

from sylva import registration


@pytest.mark.parametrize("method", ["point", "plane"])
def test_icp_recovers_transform(forest, method):
    from sylva.filters import voxel_downsample

    target = voxel_downsample(forest, 0.2)
    true = registration.rotation_z(3.0) @ registration.translation(0.15, -0.1, 0.05)
    source = target.transform(np.linalg.inv(true))
    est, info = registration.icp(source, target, max_correspondence_distance=1.0, method=method,
                                 max_iterations=100)
    np.testing.assert_allclose(est, true, atol=0.02)
    assert info["rmse"] < 0.05


def test_merge_scans(forest):
    transforms = [np.eye(4), registration.translation(1, 0, 0)]
    merged = registration.merge_scans([forest, forest], transforms)
    assert len(merged) == 2 * len(forest)
    assert set(np.unique(merged.attrs["scan_id"])) == {0, 1}
