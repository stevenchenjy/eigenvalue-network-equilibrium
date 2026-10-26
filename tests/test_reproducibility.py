import numpy as np

from network_models.convergence import simulate
from network_models.eigen_analysis import spectral_summary
from network_models.matrices import four_room


def test_identical_inputs_produce_identical_outputs():
    first = simulate(four_room(.02),[1,0,0,0])
    second = simulate(four_room(.02),[1,0,0,0])
    assert first.convergence_time == second.convergence_time
    np.testing.assert_array_equal(first.distributions,second.distributions)
    np.testing.assert_array_equal(first.distances,second.distances)
    np.testing.assert_array_equal(spectral_summary(four_room(.02)).eigenvalues,
                                  spectral_summary(four_room(.02)).eigenvalues)
