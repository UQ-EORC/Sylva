# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Ray-traced voxels: neighbour priors and the Bailey triangle-facet
inclination distributions, on scenes whose answers are known."""

import numpy as np
import pytest

from sylva import Shots, voxels


def shots_to(points, origin, classification=None, tree_id=None):
    """One single-echo pulse from ``origin`` to each point."""
    points = np.asarray(points, dtype=float)
    d = points - origin
    r = np.linalg.norm(d, axis=1)
    n = len(points)
    attrs = {}
    if classification is not None:
        attrs["classification"] = np.asarray(classification, dtype=np.uint8)
    if tree_id is not None:
        attrs["tree_id"] = np.asarray(tree_id, dtype=np.int32)
    return Shots(np.tile(origin, (n, 1)), d / r[:, None], np.arange(n), np.ones(n, np.int64), r,
                 attrs)


# --------------------------------------------------------------------------- #
# neighbour priors
# --------------------------------------------------------------------------- #


def test_neighbour_priors_bring_sparse_interior_voxels_to_min_rays():
    rng = np.random.default_rng(5)
    n = 20000
    # Pulses straight down onto a floor, sparse over one corner of the plot.
    xy = rng.uniform(0, 6, (n, 2))
    keep = (xy[:, 0] > 3) | (xy[:, 1] > 3) | (rng.random(n) < 0.02)
    xy = xy[keep]
    origin = np.column_stack([xy, np.full(len(xy), 10.0)])
    s = Shots(origin, np.tile([0.0, 0.0, -1.0], (len(xy), 1)), np.arange(len(xy)),
              np.ones(len(xy), np.int64), np.full(len(xy), 9.5))
    bounds = ((0, 0, 0), (6, 6, 3))
    plain = voxels.ray_voxelize(s, 1.0, bounds)
    min_rays = 60
    topped = voxels.ray_voxelize(s, 1.0, bounds, neighbour_prior_min_rays=min_rays)
    before, after = plain.num_beams_weighted, topped.num_beams_weighted     # (z, y, x)
    interior = np.zeros(before.shape, bool)
    interior[1:-1, 1:-1, 1:-1] = True
    sparse = interior & (before > 0) & (before < min_rays)
    assert sparse.sum() >= 4, "the scene must have sparse interior voxels"
    # Each sparse interior voxel just reaches min_rays from its neighbours...
    np.testing.assert_allclose(after[sparse], min_rays, rtol=1e-5)
    # ...and everything else is left as measured.
    np.testing.assert_array_equal(after[~sparse], before[~sparse])
    # The other statistics are topped up with the same shares.
    assert np.all(topped.path_length[sparse] > plain.path_length[sparse])
    np.testing.assert_array_equal(topped.path_length[~sparse], plain.path_length[~sparse])


# --------------------------------------------------------------------------- #
# inclination distributions: PCA normals and Bailey triangle facets
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def leaf_and_wood():
    """A leaf sheet tilted 42 degrees and a vertical wooden wall, sampled every 2 cm."""
    u = 0.2 + 0.02 * np.arange(31)
    a, b = np.meshgrid(u, u)
    a, b = a.ravel(), b.ravel()
    tilt = np.radians(42.0)
    leaf = np.column_stack([0.2 + a * np.cos(tilt), b, 1.0 + a * np.sin(tilt)])
    wall = np.column_stack([np.full(a.size, 1.6), b, 0.4 + a])
    points = np.vstack([leaf, wall])
    classes = np.repeat([4, 6], [len(leaf), len(wall)])
    shots = shots_to(points, np.array([1.0, 0.5, 2.9]), classification=classes,
                     tree_id=np.full(len(points), 3))
    return voxels.ray_voxelize(shots, 0.5, ((0, 0, 0), (2, 1, 3)), leaf_classes=[4],
                               wood_classes=[6], attenuation=["fpl", "bailey"], inclination=True)


def test_pca_histograms_put_each_surface_in_its_bin(leaf_and_wood):
    iad = leaf_and_wood.tree_iad[3]
    centres = np.degrees(iad["bin_centres"])
    np.testing.assert_allclose(centres, np.arange(2.5, 90, 5.0))
    leaf_bin, wall_bin = 8, 17                             # [40, 45) and [85, 90] degrees
    assert iad["liad"][leaf_bin] == pytest.approx(1.0)
    assert iad["wiad"][wall_bin] == pytest.approx(1.0)
    assert iad["piad"][leaf_bin] == pytest.approx(0.5)
    assert iad["piad"][wall_bin] == pytest.approx(0.5)
    assert iad["leaf_hits"] == iad["wood_hits"] == 31 * 31


def test_bailey_facets_have_the_surface_inclination(leaf_and_wood):
    iad = leaf_and_wood.tree_iad[3]
    # Every facet of the sheet is inclined 42 degrees, every facet of the wall 90.
    assert iad["liad_bailey"][8] == pytest.approx(1.0)
    assert iad["wiad_bailey"][17] == pytest.approx(1.0)
    # The plant histogram weights facets by area x sin(inclination).
    w = np.array([np.sin(np.radians(42.0)), 1.0])
    np.testing.assert_allclose(iad["piad_bailey"][[8, 17]], w / w.sum(), rtol=0.02)
    # G of a facet under the vertical mean beam is |n_z| = cos(inclination).
    assert iad["bailey_g_leaf"] == pytest.approx(np.cos(np.radians(42.0)), abs=1e-9)
    assert iad["bailey_g_wood"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(leaf_and_wood["attenuation_bailey"]).any()


def test_bailey_needs_leaf_and_wood():
    pts = np.column_stack([np.linspace(0.1, 0.9, 50), np.full(50, 0.5), np.full(50, 1.0)])
    shots = shots_to(pts, np.array([0.5, 0.5, 2.9]), classification=np.full(50, 4))
    with pytest.raises(ValueError, match="bailey method needs both leaf and wood echoes"):
        voxels.ray_voxelize(shots, 0.5, ((0, 0, 0), (1, 1, 3)), leaf_classes=[4], wood_classes=[6],
                            attenuation=["bailey"])
