"""Parity cases for sylva.coreg.simulate: the virtual scans of a known plot.

The scan sampler moved into the core; these recordings were made from the
NumPy implementation it replaced, so a seed still gives the same scan.
"""

import numpy as np

from sylva.coreg import scan_plot, simulate_plot, simulate_survey

SCAN = dict(angular_step=0.02, max_range=14.0, ground_density=6.0, foliage_density=80.0,
            understorey_density=4.0)


def _plot():
    return simulate_plot(size=20.0, n_trees=12, seed=3)


def _scan_out(prefix, cloud, visible, scanner):
    return {f"{prefix}xyz": cloud.xyz, f"{prefix}label": cloud.attrs["label"],
            f"{prefix}range": cloud.attrs["range"], f"{prefix}visible": visible,
            f"{prefix}scanner": scanner}


def plot():
    p = _plot()
    return {"coeffs": p.terrain_coeffs, "stems": p.stem_positions, "dbh": p.diameters,
            "height": np.array([t.height for t in p.trees]),
            "lean": np.array([t.lean for t in p.trees]),
            "taper": np.array([t.taper for t in p.trees]),
            "crown_base": np.array([t.crown_base for t in p.trees]),
            "terrain": p.terrain(np.array([[0.0, 0.0], [5.0, -3.0], [-9.0, 8.0]]))}


def scan():
    cloud, visible, scanner = scan_plot(_plot(), np.array([1.0, -2.0]), seed=4, **SCAN)
    return _scan_out("", cloud, visible, scanner)


def scan_scanner_frame_no_occlusion():
    cloud, visible, scanner = scan_plot(_plot(), np.array([-4.0, 3.0]), seed=6, occlusion=False,
                                        world_frame=False, min_stem_returns=20, **SCAN)
    return _scan_out("", cloud, visible, scanner)


def survey():
    s = simulate_survey(n_scans=2, plot=_plot(), seed=8, **SCAN)
    out = {"positions": s.scanner_positions}
    for k, cloud in enumerate(s.clouds):
        out.update(_scan_out(f"{k}/", cloud, s.visible[k], s.true_transforms[k][:3, 3]))
        out[f"{k}/T"] = s.true_transforms[k]
    out["overlap"] = s.overlap_stems(0, 1)
    t = simulate_survey(n_scans=4, plot=_plot(), seed=8, levelled=False, **SCAN)
    out["tilted_T"] = np.array(t.true_transforms)
    return out


CASES = {"plot": plot, "scan": scan, "survey": survey,
         "scan_scanner_frame_no_occlusion": scan_scanner_frame_no_occlusion}
