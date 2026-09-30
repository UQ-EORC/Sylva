# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Leaf projection functions G against an independent numerical integration.

The reference integrates the projection of a leaf over its azimuth and then
over the leaf angle distribution with SciPy's ``quad``:

    G(θ) = ∫ f(θl) (1/2π) ∫ |cos θ cos θl + sin θ sin θl cos φ| dφ dθl

(Ross 1981), with the de Wit (1965) densities written out here and the beta
density from ``scipy.stats``.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from sylva import leaves, voxels

HALF_PI = np.pi / 2

DE_WIT = {
    "planophile": lambda t: 2 / np.pi * (1 + np.cos(2 * t)),
    "erectophile": lambda t: 2 / np.pi * (1 - np.cos(2 * t)),
    "plagiophile": lambda t: 2 / np.pi * (1 - np.cos(4 * t)),
    "extremophile": lambda t: 2 / np.pi * (1 + np.cos(4 * t)),
    "uniform": lambda t: np.full_like(np.asarray(t, dtype=float), 2 / np.pi),
    "spherical": np.sin,
}

#: Goel and Strebel's (1984) beta fits of the de Wit types, as (mu, nu).
GOEL_STREBEL = {
    "planophile": (2.770, 1.172),
    "erectophile": (1.172, 2.770),
    "plagiophile": (3.326, 3.326),
    "extremophile": (0.433, 0.433),
    "uniform": (1.0, 1.0),
    "spherical": (1.101, 1.930),
}

BEAMS = np.radians([5.0, 30.0, 57.5, 75.0, 88.0])


def kernel(theta, leaf):
    """Mean |cos| between a beam at zenith ``theta`` and a leaf normal at
    ``leaf``, over the leaf azimuth."""
    f = lambda phi: abs(np.cos(theta) * np.cos(leaf) + np.sin(theta) * np.sin(leaf) * np.cos(phi))  # noqa: E731
    # The integrand has kinks where the cosine changes sign: give them to quad.
    c = -np.cos(theta) * np.cos(leaf) / max(np.sin(theta) * np.sin(leaf), 1e-300)
    kinks = [np.arccos(c)] if abs(c) < 1 else None
    return integrate.quad(f, 0, np.pi, points=kinks, epsabs=1e-12, limit=200)[0] / np.pi


def g_reference(theta, density):
    return integrate.quad(lambda t: kernel(theta, t) * density(t), 0, HALF_PI, epsabs=1e-10,
                          limit=200)[0]


def beta_density(mu, nu):
    """Goel and Strebel's density in θl: (1 - t)^(mu-1) t^(nu-1) / B(mu, nu), t = 2θl/π."""
    return lambda t: stats.beta(nu, mu).pdf(t / HALF_PI) / HALF_PI


@pytest.mark.parametrize("name", list(GOEL_STREBEL))
def test_beta_follows_goel_and_strebel(name):
    mu, nu = GOEL_STREBEL[name]
    got = voxels.leaf_projection(BEAMS, "twoParamBeta", [mu, nu])
    want = np.array([g_reference(b, beta_density(mu, nu)) for b in BEAMS])
    # Sylva sums the kernel over 180 steps weighted by their exact probability.
    np.testing.assert_allclose(got, want, atol=2e-5)
    # Each fit reproduces the G of its de Wit type, within the accuracy of the
    # fit (under 0.001, and 0.007 for the extremophile).
    de_wit = np.array([g_reference(b, DE_WIT[name]) for b in BEAMS])
    np.testing.assert_allclose(got, de_wit, atol=0.01)


def test_the_parameter_order_matters():
    planophile = voxels.leaf_projection(BEAMS, "twoParamBeta", [2.770, 1.172])
    erectophile = voxels.leaf_projection(BEAMS, "twoParamBeta", [1.172, 2.770])
    # Horizontal leaves face a vertical beam; vertical ones a horizontal beam.
    assert planophile[0] > 0.8 and erectophile[0] < 0.45
    assert planophile[-1] < 0.3 and erectophile[-1] > 0.5


def test_a_fitted_distribution_feeds_the_voxel_projection():
    # Inclinations drawn from Goel and Strebel's planophile: t ~ Beta(nu, mu).
    rng = np.random.default_rng(0)
    mu, nu = GOEL_STREBEL["planophile"]
    incl = stats.beta(nu, mu).rvs(200_000, random_state=rng) * HALF_PI
    fit = leaves.leaf_angle_distribution(incl, inclinations=True, n_bins=45)
    np.testing.assert_allclose(fit.goel_strebel, (mu, nu), rtol=0.03)
    assert fit.goel_strebel == (fit.beta_b, fit.beta_a)
    # G of the fitted beta agrees with G of the histogram it was fitted to.
    g_beta = voxels.leaf_projection(BEAMS, "twoParamBeta", list(fit.goel_strebel))
    np.testing.assert_allclose(g_beta, fit.g(BEAMS), atol=0.01)
