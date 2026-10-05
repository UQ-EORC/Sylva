# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The command line: where each command writes when it is not told."""

import numpy as np
import pytest

from sylva import PointCloud, cli, io


def test_beside_puts_the_output_next_to_the_input():
    assert cli._beside("/data/plot.laz", "_norm") == "/data/plot_norm.laz"
    assert cli._beside("/data/plot.laz", "_trees", ".csv") == "/data/plot_trees.csv"
    assert cli._beside("plot.ply", "", ".parquet") == "plot.parquet"
    assert cli._beside("/data/plot.laz", "_meshes", "") == "/data/plot_meshes"


def test_qsm_writes_beside_its_input(single_tree, tmp_path, capsys):
    src = tmp_path / "tree.ply"
    io.write(single_tree, src)
    cli.main(["qsm", str(src)])
    assert (tmp_path / "tree_qsm.csv").exists()
    assert "total_volume_m3" in capsys.readouterr().out
    # An explicit path still wins.
    cli.main(["qsm", str(src), str(tmp_path / "elsewhere.csv")])
    assert (tmp_path / "elsewhere.csv").exists()


def _two_tree_plot():
    """Its own generator: the rng fixture is session-scoped, so drawing from
    it here would shift every later test's random numbers."""
    from conftest import make_stem

    rng = np.random.default_rng(7)

    parts, ids = [], []
    for tid, x in enumerate([0.0, 6.0], start=1):
        pts = make_stem(rng, x, 0.0, 0.15, 6.0, density=3000)
        parts.append(pts)
        ids.append(np.full(len(pts), tid))
    xyz = np.vstack(parts)
    return PointCloud(xyz, {"height": xyz[:, 2].copy(),
                            "tree_id": np.concatenate(ids).astype(np.int32)})


def test_qsm_plot_defaults_its_outputs(tmp_path, capsys):
    src = tmp_path / "plot.laz"
    io.write(_two_tree_plot(), src)
    cli.main(["qsm-plot", str(src), "--no-wood", "--cylinders", "--meshes"])
    assert (tmp_path / "plot_trees.csv").exists()
    assert sorted(p.name for p in (tmp_path / "plot_cylinders").glob("*")) == ["tree1.csv", "tree2.csv"]
    assert sorted(p.name for p in (tmp_path / "plot_meshes").glob("*")) == ["tree1.ply", "tree2.ply"]
    assert "2 QSMs" in capsys.readouterr().out


def test_qsm_plot_says_so_without_tree_ids(single_tree, tmp_path):
    src = tmp_path / "one.ply"
    io.write(single_tree, src)
    with pytest.raises(SystemExit):
        cli.main(["qsm-plot", str(src)])


def test_trees_can_still_write_to_stdout(tmp_path, capsys):
    src = tmp_path / "plot.laz"
    io.write(_two_tree_plot(), src)
    cli.main(["trees", str(src), "-o", "-"])
    out = capsys.readouterr().out
    assert out.startswith("tree_id,") and "\n" in out
    assert not list(tmp_path.glob("*_trees.csv"))
