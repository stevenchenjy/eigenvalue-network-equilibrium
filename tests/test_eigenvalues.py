import numpy as np
import pytest

from network_models.eigen_analysis import (
    sorted_eigenvalues, spectral_summary, stationary_distribution,
)
from network_models.matrices import four_room, bottleneck_network


@pytest.mark.parametrize("c",[.01,.02,.05,.1,.2,.3,.4])
def test_reference_spectrum_and_gap(c):
    spectrum = spectral_summary(four_room(c))
    np.testing.assert_allclose(np.sort(spectrum.eigenvalues.real), np.sort([1,1-2*c,0,0]), atol=1e-12)
    assert spectrum.slem == pytest.approx(1-2*c,abs=1e-12)
    assert spectrum.gap == pytest.approx(2*c,abs=1e-12)


def test_negative_and_repeated_unit_eigenvalues():
    assert spectral_summary(four_room(.9)).slem == pytest.approx(.8)
    assert spectral_summary(four_room(.9)).gap == pytest.approx(.2)
    assert spectral_summary(four_room(1)).gap == pytest.approx(0,abs=1e-12)
    assert spectral_summary(four_room(0)).unit_eigenvalue_count == 2
    assert spectral_summary(np.eye(3)).slem == pytest.approx(1)
    assert spectral_summary([[1]]).gap == pytest.approx(1)


def test_complex_eigenvalues_use_magnitude():
    # Lazy directed three-cycle has two complex eigenvalues of modulus 1/2.
    matrix = .5*np.eye(3) + .5*np.array([[0,1,0],[0,0,1],[1,0,0]])
    values = sorted_eigenvalues(matrix)
    assert np.any(np.abs(values.imag) > .1)
    assert np.all(np.diff(np.abs(values)) <= 1e-12)
    assert spectral_summary(matrix).slem == pytest.approx(.5)


def test_nonuniform_stationary_distribution_and_reversibility():
    matrix = bottleneck_network(3,.03)
    pi = stationary_distribution(matrix)
    np.testing.assert_allclose(pi@matrix,pi,atol=1e-12)
    assert np.max(pi)-np.min(pi) > 0
    flows = pi[:,None]*matrix
    np.testing.assert_allclose(flows,flows.T,atol=1e-12)


def test_stationary_distribution_exists_for_periodic_irreducible_chain():
    np.testing.assert_allclose(stationary_distribution(four_room(1)),[.25]*4,atol=1e-12)
    with pytest.raises(ValueError,match="irreducible"):
        stationary_distribution(four_room(0))


def test_near_reducible_stationary_solve_is_not_silently_trusted():
    # The structural edge is positive, but double-precision least squares can
    # give a wrong equilibrium with a tiny residual in this ill-conditioned case.
    with pytest.raises(ArithmeticError,match="ill-conditioned|rank deficient"):
        stationary_distribution(four_room(1e-14))
    np.testing.assert_allclose(stationary_distribution(four_room(.01)),[.25]*4,atol=1e-12)
