"""Pair residuals and the HTML registration report, on scans of one synthetic scene."""

import numpy as np
import pytest

from sylva import registration
from sylva.registration_report import pair_residuals, write_report


@pytest.fixture(scope="module")
def survey(forest):
    """Four scans, each a random 40 % of the scene, and the matrices that register them."""
    rng = np.random.default_rng(3)
    true = [np.eye(4), registration.translation(0.5, 0.2, 0.0),
            registration.rotation_z(2.0) @ registration.translation(-0.4, 0.3, 0.1),
            registration.translation(0.0, 0.6, 0.0)]
    clouds = [forest[rng.random(len(forest)) < 0.4].transform(np.linalg.inv(t)) for t in true]
    return clouds, true


def _by_pair(results):
    return {(r.i, r.j): r for r in results}


def test_a_good_registration_gives_small_even_residuals(survey):
    clouds, true = survey
    results = pair_residuals(clouds, true, neighbours=3)
    assert len(results) == 6                           # every pair of four scans
    assert all(r.median < 0.03 and r.overlap > 0.9 for r in results)


def test_a_misregistered_scan_stands_out_in_every_pair_it_is_in(survey):
    clouds, true = survey
    wrong = list(true)
    wrong[2] = registration.rotation_z(4.5) @ true[2]
    by = _by_pair(pair_residuals(clouds, wrong, neighbours=3))
    for (i, j), r in by.items():
        if 2 in (i, j):
            assert r.median > 0.1 and r.p90 > r.median
        else:
            assert r.median < 0.03


def test_pairs_default_to_the_nearest_scanner_positions(survey):
    clouds, true = survey
    results = pair_residuals(clouds, true, neighbours=1)
    assert 2 <= len(results) <= 4                      # one neighbour each, shared pairs merged
    assert all(r.i < r.j for r in results)
    assert [(r.i, r.j) for r in pair_residuals(clouds, true, pairs=[(0, 3)])] == [(0, 3)]


def test_clouds_in_one_frame_need_no_transforms(survey):
    clouds, true = survey
    framed = [c.transform(t) for c, t in zip(clouds, true, strict=True)]
    assert all(r.median < 0.03 for r in pair_residuals(framed, None, neighbours=3))


def test_far_apart_scans_have_no_overlap(survey):
    clouds, true = survey
    away = [c.translate(500.0, 0.0, 0.0) if k == 3 else c for k, c in enumerate(clouds)]
    r = _by_pair(pair_residuals(away, true, pairs=[(0, 3)]))[(0, 3)]
    assert np.isnan(r.median) and r.overlap == 0.0


def test_pair_residuals_check_their_inputs(survey):
    clouds, true = survey
    with pytest.raises(ValueError, match="one matrix per cloud"):
        pair_residuals(clouds, true[:2])
    with pytest.raises(ValueError, match="4x4"):
        pair_residuals(clouds, [np.eye(3)] * 4)


def test_the_report_is_one_html_file_that_names_the_bad_scan(survey, tmp_path):
    pytest.importorskip("matplotlib")
    clouds, true = survey
    wrong = list(true)
    wrong[2] = registration.rotation_z(4.5) @ true[2]
    path = write_report(tmp_path / "report.html", clouds, wrong, ["N1", "N2", "N3", "N4"],
                        neighbours=3, worst=1, title="Test <report>")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "Test &lt;report&gt;" in text                # the title is escaped
    assert text.count("data:image/png;base64,") >= 4    # map, matrix, overlay, cross sections
    assert "http://" not in text and "https://" not in text   # nothing to fetch: self-contained
    first = text.split("Pairs, worst first")[1].split("</tr>")[1]
    assert "N3" in first and "check" in first           # the worst pair involves the bad scan
    assert "0 of 6" not in text and "3 of 6" in text    # three pairs are within tolerance


def test_the_report_of_a_good_registration_has_no_pair_to_check(survey, tmp_path):
    pytest.importorskip("matplotlib")
    clouds, true = survey
    text = write_report(tmp_path / "good.html", clouds, true, neighbours=3,
                        worst=0).read_text(encoding="utf-8")
    assert "6 of 6" in text and ">check<" not in text
    assert "The worst pairs up close" not in text


def test_the_report_checks_its_inputs(survey, tmp_path):
    clouds, true = survey
    with pytest.raises(ValueError, match="at least two"):
        write_report(tmp_path / "x.html", clouds[:1], true[:1])
    with pytest.raises(ValueError, match="one label"):
        write_report(tmp_path / "x.html", clouds, true, names=["a"])
