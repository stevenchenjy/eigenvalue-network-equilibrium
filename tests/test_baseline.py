import numpy as np
import pytest

from network_models.convergence import simulate
from network_models.eigen_analysis import stationary_distribution
from network_models.matrices import four_room


def test_known_first_two_updates():
    matrix = four_room(.1)
    np.testing.assert_allclose(matrix[0], [.45,.45,.05,.05], atol=1e-15)
    first = np.array([1.,0,0,0]) @ matrix
    second = first @ matrix
    np.testing.assert_allclose(first, [.45,.45,.05,.05], atol=1e-15)
    np.testing.assert_allclose(second, [.41,.41,.09,.09], atol=1e-15)
    result = simulate(matrix,[1,0,0,0],epsilon=1e-10,max_steps=2)
    np.testing.assert_allclose(result.distributions, [[1,0,0,0],first,second],atol=1e-15)
    assert result.status == "iteration_limit" and result.convergence_time is None


def test_stationary_reference():
    matrix = four_room(.1)
    pi = stationary_distribution(matrix)
    np.testing.assert_allclose(pi, [.25]*4, atol=1e-12)
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-12)


@pytest.mark.parametrize("c,expected",[(.4,5),(.1,31),(.01,342)])
def test_prompt_software_reference_times(c,expected):
    assert simulate(four_room(c),[1,0,0,0]).convergence_time == expected
