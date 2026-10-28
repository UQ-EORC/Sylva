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


# --------------------------------------------------------------------------- #
# plot_overlay
# --------------------------------------------------------------------------- #


@pytest.fixture
def pair(forest):
    """A scan, and the same scene as delivered by a misregistered second scan."""
    from sylva.filters import voxel_downsample

    target = voxel_downsample(forest, 0.1)
    true = registration.rotation_z(2.0) @ registration.translation(0.2, -0.1, 0.05)
    return target.transform(np.linalg.inv(true)), target, true


def _medians(fig):
    """The median distances in the plan row's column titles."""
    return [float(ax.get_title().split("median ")[1].split(" m")[0]) for ax in fig.axes[:2]
            if "median" in ax.get_title()]


def test_plot_overlay_titles_give_the_misfit_before_and_after(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    fig = registration.plot_overlay(source, target, true)
    try:
        assert len(fig.axes) == 7                      # 2 x 3 panels and the colour bar
        before, after = _medians(fig)
        assert before > 0.05 and after < 0.01
        assert fig.axes[2].get_title() == "distance to target"
    finally:
        plt.close(fig)


def test_plot_overlay_flags_a_wrong_transform(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    wrong = registration.rotation_z(-2.0) @ true
    fig = registration.plot_overlay(source, target, wrong)
    try:
        assert _medians(fig)[1] > 0.05
    finally:
        plt.close(fig)


def test_plot_overlay_without_a_transform_has_no_before_column(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    fig = registration.plot_overlay(source.transform(true), target)
    try:
        assert len(fig.axes) == 5                      # 2 x 2 panels and the colour bar
        assert _medians(fig) == pytest.approx([0.0], abs=0.01)
    finally:
        plt.close(fig)


def test_plot_overlay_is_repeatable(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    a = registration.plot_overlay(source, target, true, max_points=500)
    b = registration.plot_overlay(source, target, true, max_points=500)
    try:
        for x, y in zip(a.axes[0].collections, b.axes[0].collections, strict=True):
            np.testing.assert_array_equal(x.get_offsets(), y.get_offsets())
    finally:
        plt.close(a)
        plt.close(b)


def test_plot_overlay_checks_its_inputs(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    with pytest.raises(ValueError, match="4x4"):
        registration.plot_overlay(source, target, np.eye(3))
    with pytest.warns(UserWarning, match="no target points"), \
            pytest.raises(ValueError, match="no points"):
        registration.plot_overlay(source, target, true, slab_z=1e4, centre=(1e4, 1e4), radius=1.0)
    plt.close("all")


def test_plot_overlay_needs_matplotlib(pair, monkeypatch):
    import sys

    source, target, true = pair
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="matplotlib"):
        registration.plot_overlay(source, target, true)


# --------------------------------------------------------------------------- #
# plot_sections
# --------------------------------------------------------------------------- #


def _section_medians(fig):
    """The median distances in the titles of the horizontal sections, row by row."""
    return [float(ax.get_title().split("median ")[1].split(" m")[0])
            for ax in fig.axes if "median" in ax.get_title()]


def test_plot_sections_shows_a_stem_before_and_after(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    fig = registration.plot_sections(source, target, true, centres=[(14.0, 6.0)], radius=1.0)
    try:
        assert len(fig.axes) == 4                      # plan and side, before and after
        before, after = _section_medians(fig)
        assert before > 0.05 and after < 0.02
    finally:
        plt.close(fig)


def test_plot_sections_picks_the_best_seen_objects(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    fig = registration.plot_sections(source, target, true, n=2, radius=1.0)
    try:
        assert len(fig.axes) == 8                      # two objects, four panels each
        found = [tuple(float(v) for v in ax.get_title().split(")")[0].strip("(").split(","))
                 for ax in fig.axes if "median" in ax.get_title()][::2]
        assert len(found) == 2
        assert np.hypot(found[0][0] - found[1][0], found[0][1] - found[1][1]) >= 2.0
    finally:
        plt.close(fig)


def test_plot_sections_without_a_transform_has_no_before_panels(pair):
    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    fig = registration.plot_sections(source.transform(true), target, centres=[(14.0, 6.0)])
    try:
        assert len(fig.axes) == 2
        assert _section_medians(fig)[0] < 0.02
    finally:
        plt.close(fig)


def test_plot_sections_checks_its_inputs(pair, monkeypatch):
    import sys

    plt = pytest.importorskip("matplotlib.pyplot")
    source, target, true = pair
    with pytest.raises(ValueError, match="4x4"):
        registration.plot_sections(source, target, np.eye(3))
    with pytest.raises(ValueError, match="no target points"):
        registration.plot_sections(source, target, true, centres=[(1e4, 1e4)])
    with pytest.raises(ValueError, match="no object"):
        registration.plot_sections(source, target, true, height=500.0)
    plt.close("all")
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="matplotlib"):
        registration.plot_sections(source, target, true)
