# Tests of coregistration ICP and registration evaluation.
import numpy as np
import pytest

from sylva.coreg import (
    ICPConfig,
    evaluate_registration,
    icp,
    invert,
    planar_filter,
    se3_exp,
    transform_difference,
    transform_points,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def scene(rng):
    """A ground plane plus vertical cylinders: the geometry ICP actually sees."""
    ground = np.column_stack(
        [rng.uniform(-10, 10, 12000), rng.uniform(-10, 10, 12000), rng.normal(0, 0.005, 12000)]
    )
    stems = []
    for centre in rng.uniform(-8, 8, size=(8, 2)):
        theta = rng.uniform(0, 2 * np.pi, 3000)
        radius = rng.uniform(0.12, 0.3)
        stems.append(
            np.column_stack(
                [
                    centre[0] + radius * np.cos(theta),
                    centre[1] + radius * np.sin(theta),
                    rng.uniform(0, 6, 3000),
                ]
            )
        )
    return np.vstack([ground] + stems)


@pytest.mark.parametrize("method", ["point_to_plane", "point_to_point"])
def test_icp_recovers_a_known_perturbation(scene, method):
    T = se3_exp(np.array([0.01, -0.008, 0.06, 0.25, -0.18, 0.05]))
    moved = transform_points(invert(T), scene)
    result = icp(moved, scene, config=ICPConfig(method=method))
    rotation, translation = transform_difference(result.transform, T)
    assert np.degrees(rotation) < 0.2
    assert translation < 0.02
    assert result.fitness > 0.8


def test_icp_uses_the_initial_guess(scene):
    T = se3_exp(np.array([0.0, 0.0, 0.9, 3.0, -2.0, 0.0]))
    moved = transform_points(invert(T), scene)
    # Far outside ICP's basin of convergence without a coarse alignment.
    assert transform_difference(icp(moved, scene).transform, T)[1] > 1.0
    refined = icp(moved, scene, initial=T @ se3_exp(np.array([0, 0, 0.01, 0.1, 0.1, 0.02])))
    assert transform_difference(refined.transform, T)[1] < 0.03


def test_icp_is_stable_when_already_aligned(scene):
    result = icp(scene, scene, config=ICPConfig(max_iterations=10))
    rotation, translation = transform_difference(result.transform, np.eye(4))
    assert np.degrees(rotation) < 0.02 and translation < 0.005
    assert result.fitness > 0.95


def test_icp_handles_tiny_clouds():
    result = icp(np.zeros((3, 3)), np.zeros((3, 3)))
    assert result.fitness == 0.0 and result.n_correspondences == 0


def _forest_pair(survey, i=0, j=1):
    truth = invert(survey.true_transforms[j]) @ survey.true_transforms[i]
    source = planar_filter(survey.clouds[i].xyz, min_planarity=0.35, voxel=0.05)
    target = planar_filter(survey.clouds[j].xyz, min_planarity=0.35, voxel=0.05)
    return source, target, truth


PERTURBATIONS = [
    np.eye(4),
    se3_exp(np.array([0.004, 0.006, 0.02, 0.20, -0.15, 0.03])),
    se3_exp(np.array([-0.005, 0.003, -0.025, -0.18, 0.22, -0.04])),
]


def test_icp_converges_to_the_same_answer_from_any_initialisation(small_survey):
    """Determinism is the property ICP actually guarantees.

    It is a stronger and more honest check than agreement with ground truth:
    on sparse forest data the point-to-plane optimum is genuinely offset from
    the truth (see the next test), but a correct solver must still land on that
    optimum from every reasonable starting point.
    """
    source, target, truth = _forest_pair(small_survey)
    results = [icp(source, target, p @ truth).transform for p in PERTURBATIONS]
    for a in results:
        for b in results:
            assert transform_difference(a, b)[1] < 0.005
        assert transform_difference(a, truth)[1] < 0.25, "diverged"


def test_icp_optimum_beats_the_truth_on_its_own_objective(small_survey):
    """ICP minimises surface distance, which is not the same as being right.

    Two scans see opposite sides of every trunk, so the surface-alignment
    optimum sits a little away from the true pose.  Documenting this as a test
    keeps the pipeline honest: it is why `CoregConfig.stem_agreement_tolerance`
    exists to fall back to the (unbiased) stem solution.
    """
    source, target, truth = _forest_pair(small_survey)
    refined = icp(source, target, truth)
    truth_fitness, _, _ = evaluate_registration(source, target, truth, threshold=0.10, voxel=None)
    icp_fitness, _, _ = evaluate_registration(
        source, target, refined.transform, threshold=0.10, voxel=None
    )
    assert icp_fitness >= truth_fitness


def test_icp_accuracy_on_well_sampled_scans(survey):
    """With realistic point density the bias above shrinks to centimetres."""
    for i, j in ((0, 1), (1, 2)):
        source, target, truth = _forest_pair(survey, i, j)
        result = icp(source, target, truth)
        assert transform_difference(result.transform, truth)[1] < 0.06
        assert np.degrees(transform_difference(result.transform, truth)[0]) < 0.5


def test_evaluate_registration_scores_alignment(scene):
    fitness, rmse, n = evaluate_registration(scene, scene, np.eye(4), threshold=0.05)
    assert fitness > 0.95 and rmse < 0.01 and n > 0

    shifted = se3_exp(np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]))
    poor_fitness, _, _ = evaluate_registration(scene, scene, shifted, threshold=0.05)
    assert poor_fitness < fitness


def test_evaluate_registration_on_empty():
    fitness, _, n = evaluate_registration(np.zeros((0, 3)), np.zeros((5, 3)), np.eye(4))
    assert fitness == 0.0 and n == 0


def test_max_distances_must_match_levels():
    with pytest.raises(ValueError, match="one entry per voxel size"):
        ICPConfig(voxel_sizes=(0.2, 0.1), max_distances=(1.0,)).distances()
