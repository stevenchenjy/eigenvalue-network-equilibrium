"""Deterministic row-distribution propagation with explicit stopping status.

The measured time is the first t>=0 for which the chosen distance is strictly
below epsilon. It is for one supplied starting distribution, not a formal
worst-case mixing time. Truncated trajectories never receive an invented time.
"""
from dataclasses import dataclass

import numpy as np

from .diagnostics import (
    diagnose_chain, integer_parameter, real_array, real_scalar, validate_stochastic,
)
from .eigen_analysis import stationary_distribution


def validate_distribution(distribution, n, atol=1e-12):
    atol = real_scalar(atol, "atol", 0)
    distribution = real_array(distribution, "distribution")
    if distribution.shape != (n,):
        raise ValueError(f"distribution must be a one-dimensional vector of length {n}")
    if np.any(distribution < 0) or np.any(distribution > 1):
        raise ValueError("distribution entries must lie between 0 and 1")
    if not np.isclose(distribution.sum(),1,atol=atol,rtol=0):
        raise ValueError("distribution must sum to 1")
    return distribution


def l1_distance(p, q):
    p = real_array(p, "p")
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("p must be a nonempty distribution vector")
    p = validate_distribution(p,len(p))
    q = validate_distribution(q,len(p))
    return float(np.abs(p-q).sum())


def total_variation_distance(p, q):
    return .5*l1_distance(p,q)


@dataclass(frozen=True)
class SimulationResult:
    distributions: np.ndarray
    distances: np.ndarray
    stationary: np.ndarray
    convergence_time: int | None
    status: str
    iterations: int
    metric: str
    epsilon: float


def simulate(matrix, p0, epsilon=.001, max_steps=100000, metric="l1"):
    matrix = validate_stochastic(matrix)
    p0 = validate_distribution(p0,len(matrix))
    epsilon = real_scalar(epsilon, "epsilon", 0)
    if epsilon == 0:
        raise ValueError("epsilon must be strictly positive")
    max_steps = integer_parameter(max_steps, "max_steps", 0)
    if metric not in ("l1","tv"):
        raise ValueError("metric must be 'l1' or 'tv'")
    diagnostics = diagnose_chain(matrix)
    if not diagnostics.ergodic:
        raise ValueError(f"main simulation requires an ergodic chain (irreducible={diagnostics.irreducible}, period={diagnostics.period}); inspect diagnostics separately")
    pi = stationary_distribution(matrix)
    factor = 1. if metric == "l1" else .5
    distributions = [p0]
    distances = [factor*float(np.abs(p0-pi).sum())]
    convergence_time = 0 if distances[0] < epsilon else None
    for step in range(1,max_steps+1):
        if convergence_time is not None:
            break
        current = distributions[-1]@matrix
        if not np.all(np.isfinite(current)):
            raise ArithmeticError("nonfinite probability encountered during propagation")
        # Do not renormalize each update: retain the actual p(t)P trajectory.
        # Diagnose excessive mass drift rather than presenting a false result.
        if abs(current.sum()-1) > 1e-8 or np.min(current) < 0:
            raise ArithmeticError("probability conservation failed during propagation")
        distributions.append(current)
        distances.append(factor*float(np.abs(current-pi).sum()))
        if distances[-1] < epsilon:
            convergence_time = step
    return SimulationResult(np.array(distributions), np.array(distances),pi,
                            convergence_time,
                            "converged" if convergence_time is not None else "iteration_limit",
                            len(distances)-1,metric,epsilon)


def measure_convergence_time(matrix, p0, **kwargs):
    """Return first-passage time, or None when the configured limit is reached."""
    return simulate(matrix,p0,**kwargs).convergence_time
