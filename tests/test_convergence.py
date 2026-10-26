import numpy as np
import pytest

from network_models.convergence import l1_distance, simulate, total_variation_distance, validate_distribution
from network_models.matrices import four_room


def test_starting_distributions_change_time_without_changing_matrix():
    matrix = four_room(.1)
    times = [simulate(matrix,p).convergence_time for p in
             [[1,0,0,0],[0,1,0,0],[.5,0,.5,0],[.25]*4]]
    assert times == [31,31,1,0]


def test_strict_threshold_and_t0():
    matrix = four_room(.1)
    # Use the actual floating-point distance: the solved pi can differ from
    # exact uniform by roundoff. Equality to the computed value must not stop.
    initial = simulate(matrix,[1,0,0,0],max_steps=0).distances[0]
    result = simulate(matrix,[1,0,0,0],epsilon=initial)
    assert result.convergence_time == 1
    assert simulate(matrix,[1,0,0,0],epsilon=1.50001).convergence_time == 0
    assert simulate(matrix,[.25]*4).convergence_time == 0


def test_tv_relationship_and_consistent_stopping():
    p,q = [1,0,0,0],[.25]*4
    assert l1_distance(p,q) == 1.5
    assert total_variation_distance(p,q) == .75
    l1 = simulate(four_room(.1),p,epsilon=.002,metric="l1")
    tv = simulate(four_room(.1),p,epsilon=.001,metric="tv")
    assert l1.convergence_time == tv.convergence_time
    np.testing.assert_allclose(l1.distances,2*tv.distances,atol=1e-14)


@pytest.mark.parametrize("epsilon",[.1,.01,.001,1e-6])
def test_first_passage_and_probability_conservation(epsilon):
    result = simulate(four_room(.1),[1,0,0,0],epsilon=epsilon)
    assert result.status == "converged"
    assert result.distances[-1] < epsilon
    assert result.distances[-2] >= epsilon
    np.testing.assert_allclose(result.distributions.sum(axis=1),1,atol=1e-12)
    assert np.min(result.distributions) >= 0
    assert np.all(np.diff(result.distances) <= 1e-12)


@pytest.mark.parametrize("epsilon",[0,-1,np.nan,np.inf,True])
def test_invalid_thresholds(epsilon):
    with pytest.raises(ValueError):
        simulate(four_room(.1),[1,0,0,0],epsilon=epsilon)


@pytest.mark.parametrize("p0",[[1,0],[-.1,1.1,0,0],[1,1,0,0],[np.nan,0,0,1],[[1,0,0,0]]])
def test_invalid_distributions(p0):
    with pytest.raises(ValueError):
        simulate(four_room(.1),p0)


@pytest.mark.parametrize("limit",[-1,1.5,True,np.inf])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        simulate(four_room(.1),[1,0,0,0],max_steps=limit)


def test_nonergodic_not_labeled_convergent():
    for c in [0,1]:
        with pytest.raises(ValueError,match="ergodic"):
            simulate(four_room(c),[1,0,0,0])


def test_zero_iteration_limit_reports_truncation_and_bad_metric():
    result = simulate(four_room(.1),[1,0,0,0],max_steps=0)
    assert result.status == "iteration_limit" and result.convergence_time is None
    assert result.iterations == 0
    with pytest.raises(ValueError):
        simulate(four_room(.1),[1,0,0,0],metric="euclidean")


@pytest.mark.parametrize("atol",[np.inf,np.nan,True,-1])
def test_distribution_tolerance_is_validated(atol):
    with pytest.raises(ValueError):
        validate_distribution([1,1],2,atol=atol)


def test_ill_conditioned_target_is_not_used_for_a_convergence_claim():
    with pytest.raises(ArithmeticError,match="ill-conditioned|rank deficient"):
        simulate(four_room(1e-14),[.25]*4,max_steps=0)
