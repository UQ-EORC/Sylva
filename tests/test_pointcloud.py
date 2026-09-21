import numpy as np
import pytest

from sylva import PointCloud


def test_construct_and_index():
    pc = PointCloud(np.arange(12).reshape(4, 3), {"a": np.array([1, 2, 3, 4])})
    assert len(pc) == 4
    sub = pc[np.array([True, False, True, False])]
    assert len(sub) == 2
    assert list(sub.attrs["a"]) == [1, 3]
    assert pc[1:3].xyz.shape == (2, 3)


def test_bad_shapes():
    with pytest.raises(ValueError):
        PointCloud(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        PointCloud(np.zeros((3, 3)), {"a": np.zeros(2)})


def test_transform_and_concat():
    pc = PointCloud(np.eye(3), {"a": np.arange(3)})
    m = np.eye(4)
    m[:3, 3] = [1, 2, 3]
    moved = pc.transform(m)
    np.testing.assert_allclose(moved.xyz, np.eye(3) + [1, 2, 3])
    both = PointCloud.concatenate([pc, moved.with_attrs(b=np.zeros(3))])
    assert len(both) == 6 and set(both.attrs) == {"a"}
