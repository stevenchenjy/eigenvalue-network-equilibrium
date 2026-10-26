"""Exploratory software validation, not student-authored mathematical evidence."""
import numpy as np
import pytest

from network_models.diagnostics import diagnose_chain, validate_stochastic
from network_models.matrices import (
    asymmetric_network, bottleneck_network, four_room, hub_spoke,
    normalize_weights, two_cluster, uniform_network,
)


@pytest.mark.parametrize("c", [0, .01, .02, .05, .1, .2, .3, .4, .5, .75, 1])
def test_baseline_probability_rules(c):
    matrix = four_room(c)
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-14)
    assert np.all((matrix >= 0) & (matrix <= 1))
    np.testing.assert_array_equal(validate_stochastic(matrix), matrix)


@pytest.mark.parametrize("matrix", [
    [[.2, .3], [.5, .5]], [[-.1, 1.1], [.5, .5]],
    [[np.nan, 0], [0, 1]], [[np.inf, 0], [0, 1]],
    [[1, 0, 0], [0, 1, 0]], [], [1, 0], [[1+1e-13, -1e-13], [0, 1]],
    [[1+0j, 0], [0, 1]],
])
def test_invalid_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        validate_stochastic(matrix)


@pytest.mark.parametrize("c", [-.1, 1.1, np.nan, np.inf, True, 1+0j])
def test_invalid_baseline_parameters(c):
    with pytest.raises(ValueError):
        four_room(c)


def test_diagnostics_distinguish_structure_from_eigenvalues():
    assert diagnose_chain(four_room(.1)).ergodic
    periodic = diagnose_chain(four_room(1))
    assert periodic.irreducible and not periodic.aperiodic and periodic.period == 2
    disconnected = diagnose_chain(four_room(0))
    assert not disconnected.irreducible and not disconnected.ergodic
    # A tiny but positive edge is structurally present; do not threshold it away.
    assert diagnose_chain(four_room(1e-14)).irreducible


def test_period_three_and_single_state():
    assert diagnose_chain([[0,1,0],[0,0,1],[1,0,0]]).period == 3
    assert diagnose_chain([[1]]).period == 1
    assert diagnose_chain([[1]]).ergodic


@pytest.mark.parametrize("matrix", [
    uniform_network(4), two_cluster(3,.1), bottleneck_network(3,.03),
    hub_spoke(5,.5), asymmetric_network(5,.5,.8),
])
def test_alternative_families_are_ergodic(matrix):
    validate_stochastic(matrix)
    assert diagnose_chain(matrix).ergodic


def test_weight_normalization_rejects_empty_rows():
    with pytest.raises(ValueError):
        normalize_weights([[0,0],[1,1]])
    np.testing.assert_allclose(normalize_weights([[1,3],[2,2]]), [[.25,.75],[.5,.5]])


@pytest.mark.parametrize("constructor,kwargs", [
    (uniform_network,{"n": True}), (uniform_network,{"n": 0}),
    (two_cluster,{"n_per_group": 2.5}), (two_cluster,{"c": -1}),
    (bottleneck_network,{"bridge_weight": 0}), (bottleneck_network,{"self_loop": -1}),
    (hub_spoke,{"n": 1}), (hub_spoke,{"laziness": float("nan")}),
    (asymmetric_network,{"bias": 1.1}),
])
def test_family_parameter_validation(constructor,kwargs):
    with pytest.raises(ValueError):
        constructor(**kwargs)
