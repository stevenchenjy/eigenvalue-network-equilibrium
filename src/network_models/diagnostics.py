"""Strict input checks and graph diagnostics for finite row-stochastic chains.

Probabilities are never silently clipped or renormalized by validation. Row-sum
tolerance is absolute (rtol=0). Structural edges are entries strictly > 0, with
no numerical cutoff: a small positive edge is still a mathematical connection.
"""
from collections import deque
from dataclasses import dataclass
from math import gcd
from numbers import Real

import numpy as np


def real_scalar(value, name, minimum=None, maximum=None):
    """Reject booleans, complex numbers and nonfinite real parameters."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def integer_parameter(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def real_array(values, name):
    """Return an owned float array; never discard complex components."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real numbers")
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a rectangular real numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_stochastic(matrix, atol=1e-12):
    """Validate P[i,j]=Pr(next=j | current=i), returning a defensive copy.

    Negative entries, including tiny negative entries, are rejected. Small
    row-sum roundoff is accepted within atol, but input values are not changed.
    """
    atol = real_scalar(atol, "atol", minimum=0)
    matrix = real_array(matrix, "transition matrix")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition matrix must be a nonempty square matrix")
    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("transition probabilities must lie between 0 and 1")
    if not np.allclose(matrix.sum(axis=1), 1, atol=atol, rtol=0):
        raise ValueError("every transition-matrix row must sum to 1")
    return matrix


def _reachable(adjacency, start=0):
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


@dataclass(frozen=True)
class ChainDiagnostics:
    n: int
    irreducible: bool
    aperiodic: bool
    period: int | None

    @property
    def ergodic(self):
        """Main-study convention: finite, irreducible and aperiodic."""
        return self.irreducible and self.aperiodic


def diagnose_chain(matrix):
    """Check strong connectivity and the gcd of directed cycle lengths.

    For an irreducible graph, BFS depths d give the period as the gcd of
    d[u]+1-d[v] over all directed edges. Reducible graphs have no single
    reported period here (None); aperiodic=False means the main-study gate
    is not established, rather than asserting every component is periodic.
    """
    matrix = validate_stochastic(matrix)
    n = len(matrix)
    adjacency = matrix > 0
    irreducible = len(_reachable(adjacency)) == n and len(_reachable(adjacency.T)) == n
    if not irreducible:
        return ChainDiagnostics(n, False, False, None)
    depths = [-1] * n
    depths[0] = 0
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if depths[neighbor] == -1:
                depths[neighbor] = depths[node] + 1
                queue.append(neighbor)
    period = 0
    for node, neighbor in zip(*np.nonzero(adjacency)):
        period = gcd(period, abs(depths[node] + 1 - depths[neighbor]))
    return ChainDiagnostics(n, True, period == 1, period)
