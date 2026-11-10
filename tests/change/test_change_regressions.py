"""Regressions in sylva.change: tilted alignments, unknown vertical sigma, tree ids and the CLI."""

import argparse

import numpy as np
import pytest

from sylva import PointCloud, change, qsm
from sylva.change import EpochAlignment
from sylva.change.als import cli as change_cli
from sylva.coreg import invert, se3_exp, transform_points
from sylva.trees import Tree


def _alignment(transform=None, centre=(0.0, 0.0, 0.0), sigma_vertical=0.004, stable="stems+ground"):
    return EpochAlignment(
        transform=np.eye(4) if transform is None else transform, registration_sigma=0.005,
        sigma_horizontal=0.003, sigma_vertical=sigma_vertical, sigma_xyz=np.full(3, 0.003),
        covariance=np.eye(6) * 1e-6, centre=np.asarray(centre, dtype=float),
        stem_pairs=np.zeros((0, 2), dtype=int), stem_residuals=np.zeros((0, 2)),
        ground_residuals=np.zeros(0), stem_sigma=0.003, ground_sigma=0.01, stable=stable)


def _tilted_epochs(elevation=600.0, tilt=1e-3):
    """Stems of one plot at ``elevation`` m in two frames that differ by a
    tilt of ``tilt`` rad about x and a small shift: ``(xyz_a, xyz_b, T)``
    with ``T`` taking frame b onto frame a."""
    rng = np.random.default_rng(4)
    xyz_a = np.column_stack([rng.uniform(0, 40, 12), rng.uniform(0, 40, 12),
                             elevation + rng.uniform(-1, 1, 12)])
    T = se3_exp(np.array([tilt, 0.0, 0.0, 0.3, -0.2, 0.1]))
    return xyz_a, transform_points(invert(T), xyz_a), T


def test_match_trees_moves_each_stem_at_its_elevation():
    """A tilt of 1 mrad shifts a stem at 600 m by 0.6 m unless it is moved at its own height."""
    xyz_a, xyz_b, T = _tilted_epochs()
    a = {"x": xyz_a[:, 0], "y": xyz_a[:, 1], "dbh": np.full(12, 0.3)}
    b = {"x": xyz_b[:, 0], "y": xyz_b[:, 1], "z": xyz_b[:, 2], "dbh": np.full(12, 0.3)}
    m = change.match_trees(a, b, transform=T)
    assert len(m.pairs) == 12 and np.array_equal(m.pairs[:, 0], m.pairs[:, 1])
    assert m.distance.max() < 1e-6
    # Stems that carry their elevation (coregistration stems) the same way.
    from sylva.coreg import Stem
    stems_b = [Stem(x, y, z, 0.3) for x, y, z in xyz_b]
    stems_a = [Stem(x, y, z, 0.3) for x, y, z in xyz_a]
    assert change.match_trees(stems_a, stems_b, transform=T).distance.max() < 1e-6


def test_match_trees_without_elevations_uses_the_alignment_centre():
    xyz_a, xyz_b, T = _tilted_epochs()
    trees_a = [Tree(k + 1, x, y, dbh=0.3) for k, (x, y, _) in enumerate(xyz_a)]
    trees_b = [Tree(k + 1, x, y, dbh=0.3) for k, (x, y, _) in enumerate(xyz_b)]
    al = _alignment(T, centre=xyz_a.mean(axis=0))
    m = change.match_trees(trees_a, trees_b, transform=al)
    # Stems within 1 m of the centre's height: the tilt leaves at most a millimetre.
    assert len(m.pairs) == 12 and m.distance.max() < 2e-3
    # Under a plain matrix the stems are moved at z = 0, as documented: 0.6 m off.
    plain = change.match_trees(trees_a, trees_b, transform=T)
    assert plain.distance.min() > 0.5


def _tiny_epoch():
    rng = np.random.default_rng(1)
    n = 400
    ang = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(0, 6, n)
    xyz = np.column_stack([5 + 0.15 * np.cos(ang), 5 + 0.15 * np.sin(ang), z])
    return PointCloud(xyz, {"height": z.copy()}), np.ones(n, dtype=np.int64), [Tree(1, 5.0, 5.0, dbh=0.3)]


def test_tree_increments_refuses_an_unknown_vertical_sigma():
    """Stems alone leave z free and its sigma NaN; that is not a sigma of zero."""
    cloud, labels, trees = _tiny_epoch()
    al = _alignment(sigma_vertical=float("nan"), stable="stems")
    m = change.match_trees(trees, trees, transform=al)
    with pytest.raises(ValueError, match="give registration_sigma"):
        change.tree_increments(m, cloud, cloud, labels, labels)
    with pytest.raises(ValueError, match="give registration_sigma"):
        change.tree_increments(m, cloud, cloud, labels, labels, registration_sigma=al)
    # Given the number, or an alignment that knows it, the increments are measured.
    inc = change.tree_increments(m, cloud, cloud, labels, labels, registration_sigma=0.01)
    assert inc.settings["registration_sigma"] == 0.01 and len(inc) == 1
    known = change.match_trees(trees, trees, transform=_alignment(sigma_vertical=0.004))
    assert change.tree_increments(known, cloud, cloud, labels, labels).settings["registration_sigma"] == 0.004


def _stem_model(dr=0.0):
    rows = [[0, 0, k * 0.1, 0, 0, 1, 0.1, 0.2 - 0.01 * (k * 0.1 + 0.05) + dr, k - 1, 0, 0, 80]
            for k in range(100)]
    return qsm.QSM(np.array(rows, dtype=float))


def test_tree_match_gives_id_pairs_and_plot_qsms_take_the_match():
    """TreeMatch.pairs are positions in the tree lists; models are keyed by tree id."""
    trees_a = [Tree(10, 0.0, 0.0, dbh=0.3), Tree(20, 10.0, 0.0, dbh=0.3), Tree(30, 20.0, 0.0, dbh=0.3)]
    trees_b = [Tree(7, 20.0, 0.1, dbh=0.31), Tree(8, 0.0, 0.1, dbh=0.31), Tree(9, 40.0, 0.0, dbh=0.2)]
    m = change.match_trees(trees_a, trees_b)
    assert m.pairs.tolist() == [[0, 1], [2, 0]]
    assert m.id_pairs().tolist() == [[10, 8], [30, 7]]
    assert m.death_ids() == [20] and m.recruit_ids() == [9]
    table = {"x": np.array([0.0, 10.0]), "y": np.zeros(2)}
    assert change.match_trees(table, table).id_pairs().tolist() == [[1, 1], [2, 2]]

    a, b = _stem_model(), _stem_model(dr=0.01)
    qa, qb = {10: a, 20: a, 30: a}, {7: b, 8: b, 9: b}
    plot = change.compare_plot_qsms(qa, qb, m)
    assert [(r["fate"], r["tree_id_a"], r["tree_id_b"]) for r in plot.rows] == [
        ("survivor", 10, 8), ("survivor", 30, 7), ("death", 20, None), ("recruit", None, 9)]
    assert set(plot.changes) == {(10, 8), (30, 7)}
    assert plot.rows[0]["taper_increment_m"] == pytest.approx(0.01)
    same = change.compare_plot_qsms(qa, qb, m.id_pairs(), deaths=m.death_ids(), recruits=m.recruit_ids())
    key = ("fate", "tree_id_a", "tree_id_b", "volume_a_m3", "volume_b_m3", "change_m3")
    assert [[r[k] for k in key] for r in same.rows] == [[r[k] for k in key] for r in plot.rows]
    # Explicit lists replace the match's; pairs alone still mean no deaths or recruits.
    assert len(change.compare_plot_qsms(qa, qb, m, deaths=[], recruits=[])) == 2
    assert len(change.compare_plot_qsms(qa, qb, [(10, 8)])) == 1


def test_als_tree_change_harmonises_beside_its_output(tmp_path, monkeypatch):
    """The thinned copies of two surveys do not belong in the system's temporary directory."""
    made = {}

    class Stop(Exception):
        pass

    def temporary_directory(*a, **kw):
        made.update(kw)
        raise Stop

    monkeypatch.setattr("sylva.als.catalog", lambda path, pattern=None: object())
    monkeypatch.setattr("tempfile.TemporaryDirectory", temporary_directory)
    out = tmp_path / "results" / "trees.csv"
    args = argparse.Namespace(a="a", b="b", output=str(out), pattern="*.laz", chunk_size=None, buffer=30.0,
                              workers=1, align=False, window_linear=None, window=5.0, method="dalponte2016",
                              resolution=0.5, hmin=2.0, max_cr=10.0, harmonise=True, density_cell=10.0)
    with pytest.raises(Stop):
        change_cli._cmd_als_tree_change(args, None)
    assert made["dir"] == out.parent.resolve() and out.parent.is_dir()
