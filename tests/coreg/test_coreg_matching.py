# Tests of stem-map matching.
import numpy as np
import pytest

from sylva.coreg import (
    MatchConfig,
    Stem,
    StemMap,
    detect_stems,
    fit_ground,
    invert,
    match_stem_maps,
    stem_map_from_arrays,
    transform_difference,
    yaw_transform,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _random_stem_map(n, rng, extent=25.0, name=""):
    xy = rng.uniform(-extent, extent, size=(n, 2))
    dbh = rng.uniform(0.1, 0.6, size=n)
    return stem_map_from_arrays(xy, dbh, name=name)


def test_matches_a_pure_rotation(rng):
    source = _random_stem_map(30, rng)
    T = yaw_transform(2.1, 8.0, -5.0, 0.3)
    target = source.transformed(T)
    result = match_stem_maps(source, target)
    assert result.success
    assert result.n_inliers == 30
    rotation, translation = transform_difference(result.transform, T)
    assert rotation < 1e-6 and translation < 1e-6


def test_matches_with_partial_overlap_and_clutter(rng):
    """Half the stems are shared; the rest are scan-specific clutter."""
    shared = _random_stem_map(18, rng)
    T = yaw_transform(-1.2, -4.0, 7.0, -0.2)

    source_only = _random_stem_map(12, rng)
    target_only = _random_stem_map(12, rng)
    source = stem_map_from_arrays(
        np.vstack([shared.xy, source_only.xy]),
        np.concatenate([shared.diameters, source_only.diameters]),
    )
    moved = shared.transformed(T)
    target = stem_map_from_arrays(
        np.vstack([moved.xy, target_only.xy]),
        np.concatenate([shared.diameters, target_only.diameters]),
        z=np.concatenate([moved.positions[:, 2], target_only.positions[:, 2]]),
    )
    result = match_stem_maps(source, target)
    assert result.success
    assert result.n_inliers >= 15
    rotation, translation = transform_difference(result.transform, T)
    assert rotation < 1e-4 and translation < 0.02


def test_matching_tolerates_position_noise(rng):
    source = _random_stem_map(28, rng)
    T = yaw_transform(0.8, 3.0, 3.0, 0.0)
    moved = source.transformed(T)
    noisy = stem_map_from_arrays(moved.xy + rng.normal(0, 0.03, moved.xy.shape), moved.diameters)
    result = match_stem_maps(source, noisy)
    assert result.success and result.n_inliers >= 20
    assert transform_difference(result.transform, T)[0] < 0.01


def test_no_match_between_unrelated_maps(rng):
    a = _random_stem_map(25, rng)
    b = _random_stem_map(25, rng)
    result = match_stem_maps(a, b, MatchConfig(min_inliers=8))
    assert not result.success


def test_empty_map_returns_failure(rng):
    result = match_stem_maps(stem_map_from_arrays(np.zeros((0, 2))), _random_stem_map(10, rng))
    assert not result.success
    assert np.allclose(result.transform, np.eye(4))


def test_one_to_one_correspondences(rng):
    source = _random_stem_map(20, rng)
    target = source.transformed(yaw_transform(0.5, 1.0, 1.0, 0.0))
    result = match_stem_maps(source, target)
    assert len(set(result.correspondences[:, 1].tolist())) == len(result.correspondences)


def test_correspondences_index_the_maps_as_given(rng):
    """Sylva's correspondences index the stem maps passed in, not their top-N."""
    source = _random_stem_map(20, rng)
    T = yaw_transform(0.5, 1.0, 1.0, 0.0)
    target = source.transformed(T)
    # Shuffle the target so that index i no longer means the same tree.
    order = rng.permutation(len(target))
    target = StemMap([target[k] for k in order])
    result = match_stem_maps(source, target)
    assert result.success
    s, t = result.correspondences[:, 0], result.correspondences[:, 1]
    assert np.allclose(source.transformed(T).positions[s], target.positions[t], atol=1e-6)


def test_matching_recovers_simulated_scan_poses(survey):
    """The end-to-end coarse stage: real stem maps, no initial guess."""
    maps = [
        detect_stems(cloud.xyz, fit_ground(cloud.xyz, 0.5), name=f"scan_{k:02d}")
        for k, cloud in enumerate(survey.clouds)
    ]
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            result = match_stem_maps(maps[i], maps[j])
            expected = invert(survey.true_transforms[j]) @ survey.true_transforms[i]
            rotation, translation = transform_difference(result.transform, expected)
            assert result.success, f"pair {i}-{j} failed to match"
            assert np.degrees(rotation) < 0.5, f"pair {i}-{j}: yaw error {np.degrees(rotation)}"
            assert translation < 0.30, f"pair {i}-{j}: translation error {translation}"


def test_diameters_can_be_ignored(rng):
    """Diameters only gate candidates; geometry alone must still solve the match."""
    source = _random_stem_map(26, rng)
    T = yaw_transform(1.5, -6.0, 2.0, 0.0)
    moved = source.transformed(T)
    scrambled = stem_map_from_arrays(moved.xy, rng.uniform(0.1, 0.6, len(moved)))

    gated = match_stem_maps(source, scrambled)
    ungated = match_stem_maps(source, scrambled, MatchConfig(use_diameters=False))

    # Both recover the transform; the gate just discards correct pairs whose
    # diameters no longer agree, so it finds strictly fewer of them.
    assert transform_difference(gated.transform, T)[0] < 1e-5
    assert transform_difference(ungated.transform, T)[0] < 1e-5
    assert ungated.n_inliers == 26
    assert gated.n_inliers < ungated.n_inliers


def _stem(x, y, dbh=0.24):
    return Stem(
        x=x,
        y=y,
        z=0.0,
        dbh=dbh,
        axis=np.array([0.0, 0.0, 1.0]),
        reference_height=1.3,
        n_slices=6,
        n_points=400,
        rmse=0.005,
        coverage=0.5,
        lean_deg=1.0,
    )


def test_a_planted_lattice_is_reported_as_ambiguous():
    """A regular grid of stems matches itself shifted by one row; the matcher must say so."""
    a = StemMap([_stem(3.0 * i, 3.0 * j) for i in range(8) for j in range(8)], name="lattice")
    b = StemMap(
        [_stem(3.0 * i + 0.7, 3.0 * j - 0.4) for i in range(8) for j in range(8)], name="lattice"
    )
    match = match_stem_maps(a, b, MatchConfig(use_diameters=False))
    assert match.success
    assert match.ambiguity > 0.8, f"a lattice should be ambiguous, got {match.ambiguity:.2f}"

    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 24, size=(60, 2))
    ra = StemMap([_stem(x, y) for x, y in xy], name="random")
    rb = StemMap([_stem(x + 0.7, y - 0.4) for x, y in xy], name="random")
    match = match_stem_maps(ra, rb, MatchConfig(use_diameters=False))
    assert match.success
    assert match.ambiguity < 0.5, (
        f"an irregular stand should not be ambiguous, got {match.ambiguity:.2f}"
    )
