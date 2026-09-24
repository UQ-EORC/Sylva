# Ported from tlsalign's tests/test_pipeline.py, plus Sylva's extras (fixed
# scans, priors) and the checks of the old tests/test_coreg.py.
#
# Not ported: parallel_backend ("fork"/"thread") equivalence and the fork
# shared-state release (Sylva has threads only, no parallel_backend).
import dataclasses
import json

import numpy as np
import pytest
from conftest import make_stem

from sylva import PointCloud, io
from sylva.coreg import (
    CoregConfig,
    PoseGraph,
    Reflector,
    StemMap,
    coregister,
    coregister_prepared,
    invert,
    merge_clouds,
    place_from_prior,
    prepare_scan,
    refine_joint,
    register_pair,
    scan_plot,
    se3_exp,
    simulate_plot,
    simulate_survey,
    transform_difference,
    transform_points,
)
from sylva.coreg.pipeline import _make_logger


@pytest.fixture(scope="module")
def config():
    cfg = CoregConfig()
    cfg.verbose = False
    return cfg


@pytest.fixture(scope="module")
def registered(survey, config):
    return coregister(survey.clouds, config)


@pytest.fixture(scope="module")
def prepared(survey, config):
    return [prepare_scan(c, config, name=f"scan_{k:02d}") for k, c in enumerate(survey.clouds)]


def _pose_errors(result, survey):
    reference = invert(survey.true_transforms[result.reference])
    return [
        transform_difference(result.poses[k], reference @ survey.true_transforms[k])
        for k in range(len(result.poses))
    ]


def _pose_error(pose, k, result, survey):
    reference = invert(survey.true_transforms[result.reference])
    truth = reference @ survey.true_transforms[k]
    return transform_difference(pose, truth)[1]


def _write(path, xyz):
    io.write(PointCloud(np.asarray(xyz, dtype=float).reshape(-1, 3)), path)
    return path


def _stranger(seed, scan_seed):
    """A scan of an unrelated plot."""
    other = simulate_plot(size=30.0, n_trees=25, seed=seed)
    cloud, _, _ = scan_plot(other, np.array([0.0, 0.0]), seed=scan_seed, angular_step=0.006)
    return PointCloud(cloud.xyz)


def test_all_scans_register(registered):
    assert all(registered.registered)
    assert len(registered.successful_pairs()) == len(registered.pairs)


def test_pose_accuracy_against_ground_truth(registered, survey):
    errors = _pose_errors(registered, survey)
    for rotation, translation in errors:
        assert np.degrees(rotation) < 0.5
        assert translation < 0.10
    assert max(t for _, t in errors) < 0.10


def test_reference_scan_is_the_identity(registered):
    assert np.allclose(registered.poses[registered.reference], np.eye(4))


def test_consistency_is_reported_and_small(registered):
    consistency = registered.consistency()
    assert consistency
    for (i, j), rmse in consistency.items():
        assert rmse < 0.20, f"pair {i}-{j} disagrees by {rmse:.3f} m"


def test_report_mentions_every_scan(registered):
    report = registered.report()
    for name in registered.names:
        assert name in report
    assert "Stem agreement" in report


def test_save_roundtrip(tmp_path, registered):
    payload = json.loads(registered.save(tmp_path / "transforms.json").read_text())
    assert len(payload["scans"]) == len(registered.scans)
    for entry, pose in zip(payload["scans"], registered.poses, strict=True):
        assert np.allclose(np.array(entry["world_from_scan"]), pose)


def test_prepare_scan_produces_usable_features(survey, config):
    features = prepare_scan(survey.clouds[0], config, name="scan_00")
    assert features.n_points == len(survey.clouds[0])
    assert len(features.stem_map) > 10
    assert 0 < len(features.icp_points) < features.n_points
    assert features.ground.slope_deg >= 0


def test_prepare_scan_accepts_a_path(tmp_path, survey, config):
    path = _write(tmp_path / "scan.laz", survey.clouds[0].xyz)
    features = prepare_scan(path, config)
    assert features.name == "scan"
    assert len(features.stem_map) > 10


def test_register_pair_with_an_initial_guess(prepared, survey, config):
    a, b = prepared[0], prepared[1]
    truth = invert(survey.true_transforms[1]) @ survey.true_transforms[0]
    pair = register_pair(a, b, config, initial=truth)
    assert pair.success
    assert transform_difference(pair.transform, truth)[1] < 0.10


def test_pair_fails_cleanly_on_unrelated_scans(prepared, config):
    """Two scans with nothing in common must be reported as a failure."""
    a = prepared[0]
    b = prepare_scan(_stranger(999, 3), config, name="other")
    pair = register_pair(a, b, config)
    assert not pair.success
    assert pair.reason


def test_merge_produces_a_combined_cloud(survey, registered):
    merged = merge_clouds(survey.clouds, registered, voxel=0.10)
    assert len(merged) > 0
    total = sum(len(c) for c in survey.clouds)
    assert len(merged) < total  # overlap must collapse
    lo, hi = merged.bounds  # a property in Sylva
    assert np.all(hi - lo > 10.0)


def test_global_optimisation_can_be_disabled(survey):
    cfg = CoregConfig()
    cfg.verbose = False
    cfg.optimise_globally = False
    result = coregister(survey.clouds[:2], cfg)
    assert result.optimisation is None
    assert all(result.registered)


def test_explicit_pair_list_is_honoured(survey, config):
    result = coregister(survey.clouds, config, pairs=[(0, 1), (1, 2)])
    assert len(result.pairs) == 2
    assert all(result.registered)


def test_single_unregisterable_scan_is_flagged(survey, config):
    """A scan that matches nothing must be reported, not silently placed."""
    clouds = list(survey.clouds[:2]) + [_stranger(4242, 7)]
    result = coregister(clouds, config, names=["scan_00", "scan_01", "stranger"])
    assert result.registered[:2] == [True, True]
    assert not result.registered[2]


def test_edge_indices_map_back_to_the_right_pairs(survey, config):
    """Pose-graph edges only cover accepted pairs, so edge k is not pair k."""
    # Put the unregisterable scan first so failed and accepted pairs interleave.
    clouds = [_stranger(321, 11)] + list(survey.clouds)
    result = coregister(clouds, config)

    accepted = result.successful_pairs()
    assert len(result.edge_to_pair) == len(accepted)
    for edge_index, pair in enumerate(accepted):
        assert result.pair_for_edge(edge_index) is pair
    assert result.pair_for_edge(len(accepted)) is None
    # Any pair named as a global-optimisation outlier must be one that was accepted.
    for pair in result.rejected_pairs():
        assert pair.success


def test_screening_skips_icp_on_hopeless_pairs(survey, config):
    """Pairs rejected by stem screening must never reach ICP."""
    # Recovery is disabled here so screening is tested in isolation: it exists
    # precisely to register scans that the pairwise stage rejected, and would
    # otherwise put them back.
    cfg = dataclasses.replace(config, max_coarse_stem_rmse=0.0, recover_unregistered=False)
    result = coregister(survey.clouds, cfg)
    assert not any(p.success for p in result.pairs)
    assert all(p.icp is None for p in result.pairs)
    assert all("disagree" in p.reason or "matching failed" in p.reason for p in result.pairs)


def test_config_to_dict_covers_every_field():
    """A hand-maintained serialiser drifts; this one must track the dataclass."""
    config = CoregConfig()
    payload = config.to_dict()
    assert set(payload) == {f.name for f in dataclasses.fields(config)}
    json.dumps(payload)  # must stay JSON-serialisable
    assert payload["icp"]["voxel_sizes"][0] == config.icp.voxel_sizes[0]
    assert payload["stems"]["min_radius"] == config.stems.min_radius


def test_verbose_logger_flushes(capsys):
    _make_logger(True)("hello")
    assert capsys.readouterr().out == "hello\n"
    assert _make_logger(False)("hello") is None
    assert capsys.readouterr().out == ""


def test_threaded_matches_sequential_exactly(prepared, config):
    """Threads must be an optimisation, not a change in behaviour.

    Sylva's stand-in for tlsalign's fork/thread backend test.
    """
    sequential = coregister_prepared(prepared, dataclasses.replace(config, workers=1))
    parallel = coregister_prepared(prepared, dataclasses.replace(config, workers=4))
    assert parallel.registered == sequential.registered
    assert [p.success for p in parallel.pairs] == [p.success for p in sequential.pairs]
    assert [(p.i, p.j) for p in parallel.pairs] == [(p.i, p.j) for p in sequential.pairs]
    for a, b in zip(parallel.poses, sequential.poses, strict=True):
        assert np.allclose(a, b, atol=1e-9)


def test_saved_transforms_are_strict_json(tmp_path, survey, config):
    """NaN/Infinity are not JSON; a reader in another language must not choke."""

    def _reject(constant):
        raise AssertionError(f"non-JSON literal in output: {constant}")

    # Force pairs to fail before ICP, so the unset metrics are exercised.
    result = coregister(survey.clouds, dataclasses.replace(config, max_coarse_stem_rmse=0.0))
    text = result.save(tmp_path / "transforms.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text, parse_constant=_reject)
    assert payload["pairs"] and payload["pairs"][0]["fine_stem_rmse"] is None


def _fake_reflectors(positions):
    return [Reflector(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in positions]


def test_falls_back_to_stems_when_reflectors_mislead(prepared, config):
    """A confident but wrong target match must not cost the pair its stem match."""
    a, b = prepared[0], prepared[1]

    # Same triangle in both scans, but placed somewhere unrelated: a congruent
    # match that aligns nothing. This is the coincidence the fallback exists for.
    bogus = np.array([[0.0, 0.0, 0.0], [7.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    a = dataclasses.replace(a, reflectors=_fake_reflectors(bogus))
    b = dataclasses.replace(b, reflectors=_fake_reflectors(bogus + [40.0, 40.0, 0.0]))

    pair = register_pair(a, b, config)
    assert pair.success, "should have fallen back to stems"
    assert pair.reflector_match is None, "the bad target match must not be reported"

    # And the answer matches what stems alone would have produced.
    stems_only = register_pair(a, b, dataclasses.replace(config, use_reflectors=False))
    assert np.allclose(pair.transform, stems_only.transform, atol=1e-9)


def test_reflectors_are_used_when_they_are_right(prepared, survey, config):
    """The good case still prefers targets over stems."""
    a, b = prepared[0], prepared[1]
    truth = invert(survey.true_transforms[1]) @ survey.true_transforms[0]

    # A well-spread triangle: a thin one fixes the rotation poorly and is
    # deliberately rejected (see test_collinear_targets_are_rejected).
    corners = np.array([[0.0, 0.0, 0.0], [12.0, 0.0, 1.0], [0.0, 12.0, 0.5]])
    a = dataclasses.replace(a, reflectors=_fake_reflectors(corners))
    b = dataclasses.replace(b, reflectors=_fake_reflectors(transform_points(truth, corners)))

    pair = register_pair(a, b, config)
    assert pair.success
    assert pair.reflector_match is not None
    assert pair.reflector_match.n_inliers == 3


@pytest.fixture(scope="module")
def with_an_empty_scan(tmp_path_factory, survey, config):
    tmp = tmp_path_factory.mktemp("empty")
    empty = _write(tmp / "aborted.laz", np.zeros((0, 3)))
    good = [_write(tmp / f"s{k}.laz", c.xyz) for k, c in enumerate(survey.clouds[:2])]
    return coregister([empty] + good, config)  # reference_scan defaults to 0


def test_an_empty_scan_is_set_aside_not_fatal(with_an_empty_scan):
    """One aborted scan must not cost a survey the scans already prepared."""
    result = with_an_empty_scan
    assert result.scans[0].error, "the empty scan should carry a reason"
    assert not result.scans[0].usable
    assert not result.registered[0]
    # ... and the rest of the survey still registered.
    assert all(result.registered[1:])
    assert "SET ASIDE" in result.report()


def test_reference_moves_off_an_unusable_scan(with_an_empty_scan):
    """Anchoring the world frame to an empty scan would orphan everything."""
    result = with_an_empty_scan
    assert result.reference != 0
    assert result.scans[result.reference].usable


def test_reference_moves_into_the_largest_registered_block(survey, config):
    """A reference no accepted pair touches must not orphan a block that did register.

    Seen on a Banksia woodland survey: 17 scans registered to each other while
    the report said one scan registered, because scan 0 had no accepted pair.
    """
    result = coregister(survey.clouds, config, pairs=[(1, 2)])  # scan 0 never paired
    assert result.reference != 0
    assert all(result.registered[1:])
    assert np.allclose(result.poses[result.reference], np.eye(4))


def test_multiview_refinement_does_not_degrade_accuracy(survey, config, registered):
    """Re-aligning each scan against its neighbours must leave a good survey at least as good."""
    plain = registered
    refined = coregister(survey.clouds, dataclasses.replace(config, refine_multiview=True))
    assert all(refined.registered)
    before = max(t for _, t in _pose_errors(plain, survey))
    after = max(t for _, t in _pose_errors(refined, survey))
    assert after <= before + 0.003, (
        f"refinement made the worst scan worse: {before:.4f} -> {after:.4f}"
    )
    assert np.allclose(refined.poses[refined.reference], np.eye(4))


def test_a_tilted_scan_registers_once_levelled(tmp_path, survey, config):
    """A scan taken with the scanner on its side registers when its attitude is supplied.

    Seen on a VZ-400i survey with a tilt mount: every tilted position failed
    because the ground model and stem detector assume z is up.
    """
    # tip the last scan by 80 degrees about x and 20 about y, as a tilt mount would
    rx, ry = np.radians(80.0), np.radians(20.0)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    tilt = np.eye(4)
    tilt[:3, :3] = Ry @ Rx
    clouds = list(survey.clouds[:-1]) + [survey.clouds[-1].transform(tilt)]
    levelling = [None] * (len(clouds) - 1) + [invert(tilt)]

    blind = coregister(clouds, config)
    levelled = coregister(clouds, config, levelling=levelling)
    assert not blind.registered[-1], "the tilted scan should fail without levelling"
    assert all(levelled.registered)

    # transform_for composes the levelling out, so it applies to the raw tilted points
    ref = invert(survey.true_transforms[levelled.reference])
    for k in range(len(clouds)):
        last = k == len(clouds) - 1
        truth = ref @ survey.true_transforms[k] @ (invert(tilt) if last else np.eye(4))
        rot, trans = transform_difference(levelled.transform_for(k), truth)
        assert trans < 0.10 and np.degrees(rot) < 0.5, (
            f"scan {k}: {trans:.3f} m, {np.degrees(rot):.3f} deg"
        )
    payload = json.loads(levelled.save(tmp_path / "t.json").read_text())
    assert np.allclose(payload["scans"][-1]["levelling"], invert(tilt))


def test_an_unreadable_scan_does_not_abort_the_run(tmp_path, survey, config):
    """A corrupt file costs that file, not the hour spent on the others."""
    broken = tmp_path / "corrupt.laz"
    broken.write_bytes(b"LASF garbage that is not a point cloud")
    good = [_write(tmp_path / f"s{k}.laz", c.xyz) for k, c in enumerate(survey.clouds[:2])]

    result = coregister(good + [broken], config)
    assert result.scans[-1].error
    assert all(result.registered[:2])


def test_gnss_pre_selection_skips_distant_pairs(prepared, config):
    """Approximate positions should remove pairs that cannot possibly overlap."""
    # Place the third scan far away; the first two stay where they are.
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [500.0, 500.0, 0.0]])
    # Recovery off: it deliberately reconsiders scans that pre-selection
    # excluded, using the data rather than the fix, so it would mask the effect
    # being tested here. That interaction is checked separately below.
    cfg = dataclasses.replace(config, max_pair_distance=40.0, recover_unregistered=False)

    with_gnss = coregister_prepared(prepared, cfg, approximate_positions=positions)
    without = coregister_prepared(prepared, cfg)

    assert len(with_gnss.pairs) == 1  # only 0-1 survives
    assert len(without.pairs) == 3
    assert {(p.i, p.j) for p in with_gnss.pairs} == {(0, 1)}


def test_recovery_can_overrule_gnss_pre_selection(prepared, config):
    """Pre-selection is a speed optimisation; the data gets the final say.

    A fix that wrongly places a scan far away would otherwise exclude it for
    good, so recovery - which works from stems and ICP fitness rather than the
    fix - must still be able to bring it back.
    """
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [500.0, 500.0, 0.0]])
    cfg = dataclasses.replace(config, max_pair_distance=40.0)

    excluded = coregister_prepared(
        prepared,
        dataclasses.replace(cfg, recover_unregistered=False),
        approximate_positions=positions,
    )
    recovered = coregister_prepared(prepared, cfg, approximate_positions=positions)

    assert not excluded.registered[2], "fixture no longer excludes the third scan"
    assert recovered.registered[2], "recovery should overrule a bad fix"


def test_missing_gnss_keeps_the_pair(prepared, config):
    """An absent fix is not evidence of distance."""
    positions = np.array([[0.0, 0.0, 0.0], [500.0, 500.0, 0.0], [np.nan, np.nan, np.nan]])
    result = coregister_prepared(
        prepared,
        dataclasses.replace(config, max_pair_distance=40.0),
        approximate_positions=positions,
    )
    # 0-1 is excluded by distance; both pairs involving the unlocated scan remain.
    assert {(p.i, p.j) for p in result.pairs} == {(0, 2), (1, 2)}


@pytest.fixture(scope="module")
def sparse():
    """A survey whose scans only just reach each other, so some pairs fail."""
    plot = simulate_plot(size=50.0, n_trees=60, seed=31)
    return simulate_survey(
        n_scans=6,
        plot=plot,
        seed=31,
        angular_step=0.005,
        max_range=14.0,
        scan_radius=15.0,
    )


@pytest.fixture(scope="module")
def sparse_runs(sparse, config):
    scans = [prepare_scan(c, config, name=f"scan_{k:02d}") for k, c in enumerate(sparse.clouds)]
    without = coregister_prepared(scans, dataclasses.replace(config, recover_unregistered=False))
    with_recovery = coregister_prepared(
        scans, dataclasses.replace(config, recover_unregistered=True)
    )
    return without, with_recovery


def test_recovery_registers_scans_pairwise_matching_missed(sparse, sparse_runs):
    """A scan can overlap several registered scans without matching any one."""
    without, with_recovery = sparse_runs
    assert sum(without.registered) < len(sparse.clouds), "fixture no longer marginal"
    assert sum(with_recovery.registered) > sum(without.registered)
    assert any("recovered" in p.reason for p in with_recovery.pairs)


def test_recovery_leaves_the_already_registered_alone(sparse, sparse_runs):
    """Recovering marginal scans must not disturb the scans that were solid."""
    without, with_recovery = sparse_runs
    base = invert(sparse.true_transforms[without.reference])
    for k, was_registered in enumerate(without.registered):
        if not was_registered:
            continue
        truth = base @ sparse.true_transforms[k]
        before = transform_difference(without.poses[k], truth)[1]
        after = transform_difference(with_recovery.poses[k], truth)[1]
        assert after < before + 0.02, f"scan {k} degraded: {before:.3f} -> {after:.3f}"


def test_recovery_cannot_invent_overlap(sparse, config):
    """With nothing to match against, recovery must change nothing."""
    clouds = list(sparse.clouds[:2]) + [_stranger(777, 5)]
    result = coregister(clouds, dataclasses.replace(config, recover_unregistered=True))
    assert not result.registered[-1]


def test_joint_refinement_pulls_perturbed_poses_back(survey, registered):
    """Perturb a good solution and the joint solve must bring it back towards the truth."""
    plain = registered
    assert all(plain.registered)
    rng = np.random.default_rng(3)
    poses = []
    for k, pose in enumerate(plain.poses):
        if k == plain.reference:
            poses.append(pose.copy())
            continue
        nudge = np.r_[rng.normal(0, np.radians(0.3), 3), rng.normal(0, 0.03, 3)]
        poses.append(se3_exp(nudge) @ pose)
    edges = [(p.i, p.j) for p in plain.pairs if p.success]
    stems = [s.stem_map.positions for s in plain.scans]
    points = [s.icp_points for s in plain.scans]
    before = max(_pose_error(pose, k, plain, survey) for k, pose in enumerate(poses))
    out = refine_joint(points, poses, edges, stems, plain.reference, rounds=2, iterations=3)
    after = max(_pose_error(pose, k, plain, survey) for k, pose in enumerate(out.poses))
    assert after < 0.5 * before, f"joint refinement did not recover: {before:.4f} -> {after:.4f} m"
    assert np.allclose(out.poses[plain.reference], plain.poses[plain.reference])


def test_joint_refinement_min_voxel_points_drops_low_occupancy_patches():
    """A sparse-but-planar patch is filtered by occupancy; a dense one is not.

    The pre-existing planarity gate already rejects a truly isolated point (it
    lacks enough real neighbours to fit a plane at all), so this checks the new
    gate against something that gate cannot catch: a patch spaced wider than
    the voxel, so almost every voxel holds exactly one point, but still locally
    dense enough (within the normal-estimation radius) to look perfectly
    planar and pass every existing quality check.
    """
    import re

    def make_scan():
        xs = np.arange(-1.0, 1.001, 0.03)
        ys = np.arange(-1.0, 1.001, 0.03)
        gx, gy = np.meshgrid(xs, ys)
        dense = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        xs2 = np.arange(5.0, 7.001, 0.15)
        ys2 = np.arange(-1.0, 1.001, 0.15)
        gx2, gy2 = np.meshgrid(xs2, ys2)
        sparse = np.column_stack([gx2.ravel(), gy2.ravel(), np.zeros(gx2.size)])
        return np.vstack([dense, sparse])

    pts = make_scan()
    points = [pts, pts.copy()]
    poses = [np.eye(4), np.eye(4)]
    stems = [np.zeros((0, 3)), np.zeros((0, 3))]

    def n_correspondences(min_voxel_points):
        counts = []

        def log(msg):
            m = re.search(r"([\d,]+) correspondences", msg)
            if m:
                counts.append(int(m.group(1).replace(",", "")))

        refine_joint(
            points,
            poses,
            [(0, 1)],
            stems,
            0,
            voxel_sizes=(0.10,),
            max_distances=(0.30,),
            rounds=1,
            iterations=1,
            min_voxel_points=min_voxel_points,
            log=log,
        )
        return counts[0]

    n_default = n_correspondences(1)
    n_gated = n_correspondences(2)
    assert n_default > 0 and n_gated > 0, "the dense patch must still match in both cases"
    assert n_gated < n_default, "min_voxel_points=2 must drop the sparse-but-planar patch"


# --------------------------------------------------------------------------- #
# Sylva's extras: scans held fixed, and approximate poses as priors
# --------------------------------------------------------------------------- #


def _relative_truth(survey, k, reference=0):
    """Scan k's world_from_scan in scan ``reference``'s frame."""
    return invert(survey.true_transforms[reference]) @ survey.true_transforms[k]


def test_fixed_scans_hold_and_the_rest_register_into_their_frame(prepared, survey, config):
    """New scans join an existing project: the held poses define the world frame."""
    fixed = {0: survey.true_transforms[0], 1: survey.true_transforms[1]}
    result = coregister_prepared(prepared, config, fixed=fixed)
    assert all(result.registered)
    assert (0, 1) not in {(p.i, p.j) for p in result.pairs}, "two held scans are never paired"
    for k, pose in fixed.items():
        assert np.allclose(result.poses[k], pose)
    rot, trans = transform_difference(result.poses[2], survey.true_transforms[2])
    assert trans < 0.10 and np.degrees(rot) < 0.5, (trans, np.degrees(rot))


def test_a_pair_contradicting_the_priors_is_refused(prepared, survey, config):
    """A result that moves a scanner far from its prior is refused, however well it fits.

    The priors of scans 0 and 1 are right; scan 2's is 10 m off. The stems
    place scan 2 correctly, but the pipeline cannot know which to believe, and
    a stem match that confirms a wrong alignment in a repetitive stand looks
    exactly like this: both of scan 2's pairs must be refused.
    """
    priors = [_relative_truth(survey, k) for k in range(3)]
    priors[2] = priors[2].copy()
    priors[2][:3, 3] += [10.0, 0.0, 0.0]
    result = coregister_prepared(prepared, config, priors=priors)

    by_pair = {(p.i, p.j): p for p in result.pairs}
    assert by_pair[(0, 1)].success
    for key in ((0, 2), (1, 2)):
        assert not by_pair[key].success
        assert "refused by the prior" in by_pair[key].reason
    assert result.registered[:2] == [True, True]
    assert not result.registered[2], "neither the stems nor a 10 m wrong prior may place it"


def test_place_from_prior_corrects_height_and_refines(prepared, survey, config):
    """A scan placed from an approximate pose: metres off in height, decimetres in plan."""
    poses = [_relative_truth(survey, k) for k in range(2)]
    truth = _relative_truth(survey, 2)
    prior = se3_exp(np.array([0.0, 0.0, np.radians(1.0), 0.25, -0.20, 0.0])) @ truth
    prior[2, 3] += 2.0  # a typical GNSS height error
    result, used = place_from_prior(prepared[2], prepared[:2], poses, prior, config)
    assert result.success, result.reason
    assert set(used) == {0, 1}
    rot, trans = transform_difference(result.transform, truth)
    assert trans < 0.10 and np.degrees(rot) < 0.5, (trans, np.degrees(rot))


def test_a_scan_without_stems_is_placed_from_its_prior(prepared, survey, config):
    """Too few stems to match, so the prior places it and ICP refines it."""
    scans = list(prepared)
    scans[2] = dataclasses.replace(scans[2], stem_map=StemMap([], name=scans[2].name))
    truth = _relative_truth(survey, 2)
    prior = se3_exp(np.array([0.0, 0.0, np.radians(1.0), 0.25, -0.20, 0.0])) @ truth
    priors = [_relative_truth(survey, 0), _relative_truth(survey, 1), prior]
    result = coregister_prepared(scans, config, priors=priors)
    assert all(result.registered)
    assert any("placed from its prior" in p.reason for p in result.pairs)
    rot, trans = transform_difference(result.poses[2], truth)
    assert trans < 0.10 and np.degrees(rot) < 0.5, (trans, np.degrees(rot))


# --------------------------------------------------------------------------- #
# From the old tests/test_coreg.py: a stand of cylinders with 3-D offsets
# --------------------------------------------------------------------------- #


def _stand(seed=0, n=45, size=60.0):
    """Stems of varied diameter on a gently sloping ground, with a little low noise."""
    rng = np.random.default_rng(seed)
    xy = []
    while len(xy) < n:
        p = rng.uniform(0, size, 2)
        if all(np.hypot(*(p - q)) > 3.0 for q in xy):
            xy.append(p)
    parts = []
    for x, y in xy:
        r = rng.uniform(0.08, 0.3)
        parts.append(make_stem(rng, x, y, r, 6.0, z0=0.03 * x, density=1500, noise=0.004))
    g = rng.uniform(0, size, (120_000, 2))
    parts.append(np.column_stack([g, 0.03 * g[:, 0] + rng.normal(0, 0.01, len(g))]))
    return np.vstack(parts)


def _pose(yaw_deg, t):
    T = np.eye(4)
    a = np.radians(yaw_deg)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, 3] = t
    return T


def _err(A, B):
    rot, trans = transform_difference(A, B)
    return trans, float(np.degrees(rot))


@pytest.fixture(scope="module")
def stand(config):
    world = _stand()
    truth = [_pose(0, [0, 0, 0]), _pose(35, [22, 4, 1.5]), _pose(-80, [12, 24, -0.8])]
    centres = [(22, 22), (38, 26), (28, 40)]
    scans = []
    for k, (c, T) in enumerate(zip(centres, truth, strict=True)):
        keep = np.hypot(world[:, 0] - c[0], world[:, 1] - c[1]) < 22.0
        scans.append(prepare_scan(transform_points(invert(T), world[keep]), config, name=f"s{k}"))
    return scans, truth


def test_register_pair_recovers_a_vertical_offset(stand, config):
    scans, truth = stand
    assert all(s.usable for s in scans)
    r = register_pair(scans[1], scans[0], config, i=1, j=0)
    assert r.success, r.reason
    dt, dr = _err(r.transform, invert(truth[0]) @ truth[1])  # scan-0 frame from scan-1 frame
    assert dt < 0.03 and dr < 0.2, (dt, dr)
    assert r.n_stem_matches >= 5 and r.fitness > 0.3


def test_stand_registers_and_places_against_fixed(stand, config):
    scans, truth = stand
    res = coregister_prepared(scans, config)
    assert all(res.registered)
    for k in (1, 2):
        dt, dr = _err(res.poses[k], truth[k])
        assert dt < 0.03 and dr < 0.2, (k, dt, dr)
    # Scans 0 and 1 trusted, scan 2 registered into their frame.
    held = coregister_prepared(scans, config, fixed={0: truth[0], 1: truth[1]})
    assert held.registered[2]
    dt, dr = _err(held.poses[2], truth[2])
    assert dt < 0.03 and dr < 0.2, (dt, dr)


def test_pose_graph_spreads_loop_closure_error():
    truth = [_pose(0, [0, 0, 0]), _pose(10, [10, 0, 0]), _pose(20, [10, 10, 0])]
    g = PoseGraph(3)
    rng = np.random.default_rng(1)
    for i, j in [(1, 0), (2, 1), (2, 0)]:
        rel = invert(truth[j]) @ truth[i]
        noisy = rel @ se3_exp(np.r_[rng.normal(0, 1e-3, 3), rng.normal(0, 0.02, 3)])
        g.add_edge(i, j, noisy, rmse=0.02, fitness=0.5, n_correspondences=1000)
    g.initialise(0)
    assert g.optimise().rejected_edges == []
    for k in (1, 2):
        assert _err(g.poses[k], truth[k])[0] < 0.05
