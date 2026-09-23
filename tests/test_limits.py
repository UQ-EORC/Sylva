# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Allocations that would not fit are refused, with a message worth reading."""

import numpy as np
import pytest

import sylva
from sylva import PointCloud, limits, qsm, voxels


@pytest.fixture
def small_budget():
    """Pretend the machine has 1 GB free."""
    limits.set_budget(1.0)
    yield
    limits.set_budget(None)


def test_the_budget_can_be_set_and_put_back():
    was = limits.budget()
    limits.set_budget(2.0)
    assert limits.budget() == 2_000_000_000
    limits.set_budget(None)
    assert limits.budget() == was
    assert limits.available() is None or limits.available() > 0


def test_sizes_read_the_way_people_write_them():
    assert limits.human(512) == "512 B"
    assert limits.human(2_400_000_000) == "2.4 GB"
    assert limits.human(3.2e12) == "3.2 TB"


def test_a_grid_that_will_not_fit_says_what_it_needed(small_budget):
    with pytest.raises(ValueError) as e:
        limits.check(10**10, 64, "a 1000 x 1000 x 10000 voxel grid at 0.01 m", "a larger voxel")
    msg = str(e.value)
    assert "640.0 GB" in msg and "1.0 GB" in msg and "a larger voxel" in msg
    assert "SYLVA_MEM_BUDGET" in msg
    limits.check(1000, 64, "a small thing", "nothing")           # room for this one


def test_ray_voxelize_refuses_a_grid_it_cannot_hold(small_budget):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 30, (500, 3))
    cloud = PointCloud(xyz, {"sx": np.zeros(500), "sy": np.zeros(500), "sz": np.full(500, 40.0)})
    shots = sylva.Shots.from_ray_cloud(cloud)
    with pytest.raises(ValueError, match="voxel grid"):
        voxels.ray_voxelize(shots, 0.01, bounds=((0, 0, 0), (100, 100, 40)))
    # The same call at a size that fits still runs.
    grid = voxels.ray_voxelize(shots, 2.0, bounds=((0, 0, 0), (30, 30, 40)))
    assert grid.shape == (15, 15, 20)


def test_buttress_mesh_refuses_an_impossible_raster(small_budget):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 1, (2000, 3))
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    with pytest.raises(ValueError, match="buttress raster"):
        qsm.buttress_mesh(cloud, (0.5, 0.5), ground_z=0.0, resolution=0.001, max_radius=10.0)
