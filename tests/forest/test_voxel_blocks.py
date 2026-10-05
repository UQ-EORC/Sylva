"""Block tracing (ray_voxelize with block_size / out) against the whole-grid trace."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sylva import Shots, synthetic, voxels
from sylva.raster import Raster
from sylva.voxels.blocks import BlockedVoxelGrid, open_blocked

TREES = [(5.0, 5.0, 0.3, 9.0), (12.0, 6.0, 0.25, 7.0), (7.0, 13.0, 0.35, 10.0), (14.0, 14.0, 0.2, 6.0)]
BOUNDS = ((-1.0, -1.0, -1.5), (19.0, 19.0, 11.5))
LABELS = dict(ground_class=2, leaf_classes=[4], wood_classes=[5])


@pytest.fixture(scope="module")
def forest():
    return synthetic.forest(TREES, size=18.0, ground_points=6000, margin=2.0, seed=3)


@pytest.fixture(scope="module")
def scene(forest):
    """Two scans of the forest, misses and multi-echo pulses included."""
    parts = [synthetic.scan(forest, origin=o, resolution_deg=1.5, max_zenith_deg=125.0, max_echoes=3,
                            echo_separation=0.4) for o in ((9.0, 9.0, 1.6), (3.0, 15.0, 1.4))]
    shots = Shots.concatenate(parts)
    counts = np.diff(np.r_[shots.echo_start, shots.n_echoes])
    assert (counts == 0).any() and (counts > 1).any()
    return shots


@pytest.fixture(scope="module")
def dtm():
    x = -3.0 + 0.5 + np.arange(24.0)
    X, Y = np.meshgrid(x, x)
    z = synthetic.terrain_height(X.ravel(), Y.ravel()).reshape(X.shape)
    return Raster(z, xmin=-3.0, ymin=-3.0, resolution=1.0)


def fields_of(grid):
    return {n: grid[n] for n in grid.fields}


def assert_close_to_whole(blocked, whole):
    """Counts equal; single-precision sums equal to their rounding.

    A whole-grid trace on several threads adds to a voxel in whatever order
    the threads reach it, so its sums can differ from the (ordered) blocked
    ones in the last bit.
    """
    assert blocked.shape == whole.shape
    assert blocked.fields == whole.fields
    for n in whole.fields:
        a, b = whole[n], blocked[n]
        if a.dtype.kind in "iu":
            np.testing.assert_array_equal(b, a, err_msg=n)
        else:
            np.testing.assert_allclose(b, a, rtol=3e-7, atol=1e-12, err_msg=n)


@pytest.mark.parametrize("occlusion", [False, True])
@pytest.mark.parametrize("with_dtm", [False, True])
def test_blocks_match_the_whole_grid(scene, dtm, occlusion, with_dtm):
    kw = dict(LABELS, occlusion=occlusion, dtm=dtm if with_dtm else None, beam=(0.007, 0.00027),
              attenuation=["fpl", "ppl"], subvoxel_split=2)
    whole = voxels.ray_voxelize(scene, 0.5, BOUNDS, **kw)
    assert whole.num_hit_leaf.sum() > 0 and whole.num_hit_wood.sum() > 0
    assert (whole.num_beams_occluded.sum() > 0) == occlusion
    first = None
    for block, workers in [(7, 1), (7, 4), ((5, 11, 3), 3), (64, 2), ((40, 1, 26), None)]:
        g = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=block, workers=workers, **kw)
        assert g.block_stats["n_blocks"] == tuple(-(-s // min(b, s)) for s, b in zip(g.shape, np.broadcast_to(block, 3)))
        assert_close_to_whole(g, whole)
        # Independent of the block size and the worker count, to the bit.
        f = fields_of(g)
        if first is None:
            first = f
        for n in f:
            np.testing.assert_array_equal(f[n], first[n], err_msg=n)
    for m in ("pad_fpl", "attenuation_ppl", "transmittance", "state", "distance_from_ground"):
        np.testing.assert_allclose(g[m], whole[m], rtol=1e-5, atol=1e-9, err_msg=m)


def test_blocks_equal_a_one_thread_whole_grid_bit_for_bit(tmp_path, scene, dtm):
    # On one thread the whole-grid trace adds pulses in order, as each block does.
    path = tmp_path / "scene.parquet"
    scene.save(path, double=True, origin_tolerance=0)
    script = textwrap.dedent(f"""
        import numpy as np
        from sylva import Shots, voxels
        from sylva.raster import Raster
        s = Shots.load({str(path)!r})
        d = np.load({str(tmp_path / 'dtm.npy')!r})
        dtm = Raster(d, xmin=-3.0, ymin=-3.0, resolution=1.0)
        kw = dict(ground_class=2, leaf_classes=[4], wood_classes=[5], occlusion=True, dtm=dtm,
                  beam=(0.007, 0.00027), attenuation=["fpl", "ppl"])
        w = voxels.ray_voxelize(s, 0.5, {BOUNDS!r}, **kw)
        b = voxels.ray_voxelize(s, 0.5, {BOUNDS!r}, block_size=(6, 9, 4), workers=3, **kw)
        bad = [n for n in w.fields if not np.array_equal(w[n], b[n])]
        print("differ:", bad)
    """)
    np.save(tmp_path / "dtm.npy", dtm.data)
    env = dict(os.environ, RAYON_NUM_THREADS="1", PYTHONPATH=str(Path(voxels.__file__).parents[1]))
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert "differ: []" in out.stdout, out.stdout + out.stderr


def test_other_options_and_whole_grid_refinements(scene, dtm):
    for kw in (dict(flat_top=True, weighting="first", occlusion=True),
               dict(weighting="full", unbounded_range=6.0, attenuation="transmittance"),
               dict(neighbour_prior_min_rays=5, inclination=True, attenuation=["fpl", "bailey"])):
        kw = dict(LABELS, dtm=dtm, **kw)
        whole = voxels.ray_voxelize(scene, 0.5, BOUNDS, **kw)
        g = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=(6, 8, 5), **kw)
        assert_close_to_whole(g, whole)
        if kw.get("inclination"):
            assert g.tree_iad.keys() == whole.tree_iad.keys()
            np.testing.assert_allclose(g.lad_bailey, whole.lad_bailey, rtol=1e-5, atol=1e-9)


def test_streaming_a_shots_file(tmp_path, scene, dtm):
    path = tmp_path / "scene.parquet"
    scene.save(path, row_group_size=9000)
    kw = dict(LABELS, dtm=dtm, occlusion=True, beam=(0.007, 0.00027))
    whole = voxels.ray_voxelize(path, 0.5, BOUNDS, **kw)
    a = voxels.ray_voxelize(path, 0.5, BOUNDS, block_size=8, **kw)
    b = voxels.ray_voxelize(path, 0.5, BOUNDS, block_size=(13, 5, 7), workers=2, max_memory=0.0005, **kw)
    assert b.block_stats["n_passes"] > 5
    assert_close_to_whole(a, whole)
    for n in a.fields:
        np.testing.assert_array_equal(a[n], b[n], err_msg=n)
    # Without bounds the grid comes from the file's echo bounds, as for the whole trace.
    assert voxels.ray_voxelize(path, 0.5, block_size=16).shape == voxels.ray_voxelize(path, 0.5).shape
    with pytest.raises(ValueError):
        voxels.ray_voxelize(path, 0.5, block_size=8, ground=np.zeros(scene.n_echoes, bool))


def test_blocks_on_disk(tmp_path, scene, dtm, forest):
    kw = dict(LABELS, dtm=dtm, occlusion=True, beam=(0.007, 0.00027), attenuation=["fpl", "ppl"])
    mem = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=(8, 8, 8), **kw)
    per_voxel = mem.block_stats["bytes_per_voxel"]
    allowance = 3 * 512 * per_voxel / 1e9
    g = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=(8, 8, 8), out=tmp_path / "grid", max_memory=allowance, **kw)
    assert isinstance(g, BlockedVoxelGrid)
    st = g.block_stats
    nx, ny, nz = g.shape
    # Bounded memory: never more than three blocks' accumulators at once.
    assert st["peak_voxels"] <= 3 * 512 and st["n_passes"] >= nx * ny * nz / (3 * 512)
    assert st["peak_voxels"] * 10 < nx * ny * nz
    assert g.shape == mem.shape and g.n_blocks == (5, 5, 4) and g.block_size == (8, 8, 8)
    assert st["blocks_written"] == len(g.blocks_present()) <= 100

    same = open_blocked(tmp_path / "grid")
    for n in mem.fields:
        np.testing.assert_array_equal(same[n], mem[n], err_msg=n)
        np.testing.assert_array_equal(g.to_grid()[n], mem[n], err_msg=n)
    np.testing.assert_array_equal(g.pad_fpl, mem.pad_fpl)
    np.testing.assert_array_equal(g.observed, mem.observed)
    np.testing.assert_array_equal(g.profile("pad_fpl", 3), mem.profile("pad_fpl", 3))
    occ_b, occ_m = g.occlusion_profile(0.5), mem.occlusion_profile(0.5)
    for k in ("height", "n_voxels", "observed", "occluded", "mean_beams"):
        np.testing.assert_array_equal(occ_b[k], occ_m[k])
    assert occ_b["total"] == occ_m["total"]
    np.testing.assert_array_equal(g.observed_map(max_height=8.0), mem.observed_map(max_height=8.0))
    labels = forest.attrs["tree_id"].astype(np.int64) - 1
    ts_b, ts_m = voxels.tree_sampling(g, forest, labels), voxels.tree_sampling(mem, forest, labels)
    for k in ts_m:
        np.testing.assert_array_equal(ts_b[k], ts_m[k], err_msg=k)

    # The same .vox text, written a slab of blocks at a time.
    assert g.write(tmp_path / "b.vox") == mem.write(tmp_path / "m.vox")
    assert (tmp_path / "b.vox").read_text() == (tmp_path / "m.vox").read_text()

    # Blocks and boxes read alone.
    blk = g.block((1, 2, 0))
    assert blk.shape == (8, 8, 8)
    np.testing.assert_allclose(blk.origin, mem.origin + np.array([8, 16, 0]) * 0.5)
    np.testing.assert_array_equal(blk.num_beams, mem.num_beams[0:8, 16:24, 8:16])
    np.testing.assert_array_equal(blk.pad_fpl, mem.pad_fpl[0:8, 16:24, 8:16])
    box = g.read((3, 4, 5), (20, 9, 17))
    np.testing.assert_array_equal(box.free_path_length, mem.free_path_length[5:17, 4:9, 3:20])
    seen = [idx for idx, _ in g.blocks()]
    assert len(seen) == 100 and seen[0] == (0, 0, 0) and seen[1] == (1, 0, 0)
    assert "BlockedVoxelGrid(40x40x26" in repr(g)
    assert "pad_g0_5" in g.metrics and "num_beams" in g.fields


def test_a_grid_too_large_to_trace_whole(tmp_path, scene, monkeypatch):
    # With 2 MB to spend, the whole trace (0.23 kB a voxel, 41 600 voxels) is
    # refused; blocks written as they finish fit.
    monkeypatch.setenv("SYLVA_MEM_BUDGET", "0.002")
    with pytest.raises(ValueError, match="SYLVA_MEM_BUDGET"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, **LABELS)
    g = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=8, out=tmp_path / "g", **LABELS)
    st = g.block_stats
    assert st["peak_voxels"] * st["bytes_per_voxel"] <= 1_000_000
    monkeypatch.delenv("SYLVA_MEM_BUDGET")
    whole = voxels.ray_voxelize(scene, 0.5, BOUNDS, **LABELS)
    np.testing.assert_array_equal(g["num_hits"], whole["num_hits"])
    np.testing.assert_array_equal(g["num_beams"], whole["num_beams"])


def test_empty_and_invalid_input(tmp_path, scene):
    empty = Shots(origin=np.zeros((0, 3)), direction=np.zeros((0, 3)), echo_start=np.zeros(0, np.int64),
                  echo_count=np.zeros(0, np.uint32), echo_range=np.zeros(0))
    g = voxels.ray_voxelize(empty, 1.0, ((0, 0, 0), (4, 4, 4)), block_size=2)
    assert g.shape == (4, 4, 4) and g.num_beams.sum() == 0
    d = voxels.ray_voxelize(empty, 1.0, ((0, 0, 0), (4, 4, 4)), block_size=2, out=tmp_path / "e")
    assert d.blocks_present() == [] and d.block_stats["blocks_written"] == 0
    assert d.num_beams.sum() == 0 and np.all(d.attenuation_fpl == 0)
    with pytest.raises(ValueError, match="pass bounds"):
        voxels.ray_voxelize(empty, 1.0, block_size=2)
    for bad in (0, (1, 2), (4, -1, 4), 2.5, "8"):
        with pytest.raises(ValueError, match="block_size"):
            voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=bad)
    with pytest.raises(ValueError, match="max_memory"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=8, max_memory=-1)
    with pytest.raises(ValueError, match="max_memory"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=8, max_memory=np.nan)
    with pytest.raises(ValueError, match="workers"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=8, workers=0)
    with pytest.raises(ValueError, match="blocked trace"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, max_memory=1.0)
    with pytest.raises(ValueError, match="memory"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=16, max_memory=1e-6)
    with pytest.raises(ValueError, match="whole grid"):
        voxels.ray_voxelize(scene, 0.5, BOUNDS, out=tmp_path / "i", inclination=True, **LABELS)
    with pytest.raises(ValueError, match="bounds"):
        voxels.ray_voxelize(scene, 0.5, ((0, 0, 0), (0, 1, 1)), block_size=4)
    with pytest.raises(ValueError, match="voxel_size"):
        voxels.ray_voxelize(scene, np.nan, BOUNDS, block_size=4)
    with pytest.raises(OSError):
        open_blocked(tmp_path / "missing")
    g = voxels.ray_voxelize(scene, 0.5, BOUNDS, block_size=16, out=tmp_path / "ok")
    with pytest.raises(ValueError):
        g.block((9, 0, 0))
    with pytest.raises(ValueError):
        g.read((0, 0, 0), (0, 5, 5))
    with pytest.raises(ValueError):
        g["no_such_field"]
