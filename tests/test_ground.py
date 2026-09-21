import numpy as np
import pytest

from sylva import ground


@pytest.mark.parametrize("method", ["csf", "pmf"])
def test_ground_classification(forest, method):
    fn = ground.classify_ground_csf if method == "csf" else ground.classify_ground_pmf
    classified = fn(forest)
    mask = ground.ground_mask(classified)
    # First 20000 points are terrain, rest are vegetation.
    terrain = mask[:20000].mean()
    veg = mask[20000:].mean()
    assert terrain > 0.9, terrain
    assert veg < 0.05, veg


def test_dtm_and_normalize(forest):
    classified = ground.classify_ground_csf(forest)
    dtm = ground.make_dtm(classified, resolution=0.5)
    assert not np.isnan(dtm.data).any()
    # DTM should follow z = 0.05 x + 0.2 sin(y/3)
    xs, ys = dtm.cell_centers()
    expected = 0.05 * xs + 0.2 * np.sin(ys / 3)
    interior = (xs > 1) & (xs < 19) & (ys > 1) & (ys < 19)
    assert np.abs(dtm.data - expected)[interior].max() < 0.1

    norm = ground.normalize_height(classified, dtm)
    h = norm.attrs["height"]
    assert np.abs(h[:20000]).max() < 0.15
    assert h.max() > 14.5

    flat = ground.flatten(classified, dtm)
    np.testing.assert_allclose(flat.z, h)


def test_chm(forest):
    classified = ground.classify_ground_csf(forest)
    dtm = ground.make_dtm(classified, resolution=0.5)
    norm = ground.normalize_height(classified, dtm)
    chm = ground.make_chm(norm, resolution=0.5)
    assert chm.data.max() == pytest.approx(15.0, abs=0.3)
    assert chm.data.min() >= 0
