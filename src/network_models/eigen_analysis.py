"""Numerical spectra and left stationary vectors for exploratory study."""
from dataclasses import dataclass

import numpy as np

from .diagnostics import diagnose_chain, real_scalar, validate_stochastic


def sorted_eigenvalues(matrix):
    """Sort descending by modulus, then real and imaginary parts for ties.

    No rounding is used to discard imaginary parts or to sort by algebraic
    value. Tiny differences between BLAS/LAPACK platforms are still possible.
    """
    matrix = validate_stochastic(matrix)
    values = np.asarray(np.linalg.eigvals(matrix), dtype=complex)
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("eigenvalue calculation returned nonfinite values")
    order = np.lexsort((-values.imag, -values.real, -np.abs(values)))
    return values[order]


@dataclass(frozen=True)
class SpectralSummary:
    eigenvalues: np.ndarray
    slem: float
    gap: float
    unit_eigenvalue_count: int


def spectral_summary(matrix, eigen_atol=1e-8):
    """Remove exactly one eigenvalue nearest 1, then use the remaining moduli.

    Multiple eigenvalues near 1 are recorded, not all removed: a reducible
    chain may have SLEM=1. The singleton chain has empty nonstationary spectrum;
    by software convention its SLEM=0 and gap=1. Gap is 1-SLEM (absolute gap),
    including negative/complex eigenvalues, not always the algebraic gap.
    """
    eigen_atol = real_scalar(eigen_atol, "eigen_atol", 0)
    values = sorted_eigenvalues(matrix)
    index = int(np.argmin(np.abs(values-1)))
    if abs(values[index]-1) > eigen_atol:
        raise ArithmeticError("could not identify stationary eigenvalue 1 within tolerance")
    remaining = np.delete(values, index)
    slem = float(np.max(np.abs(remaining))) if len(remaining) else 0.
    if slem > 1+eigen_atol:
        raise ArithmeticError("calculated SLEM exceeds 1 beyond tolerance")
    slem = min(slem,1.)  # Only repair an above-1 eigensolver roundoff error.
    return SpectralSummary(values, slem, 1-slem,
                           int(np.count_nonzero(np.abs(values-1) <= eigen_atol)))


def stationary_distribution(matrix, atol=1e-10):
    """Solve (P.T-I)pi=0 with sum(pi)=1 and check the row-vector residual.

    Irreducibility is required to avoid choosing an arbitrary stationary
    distribution for a reducible model. Periodic irreducible chains still
    have a unique stationary distribution; this function does not imply
    convergence. Tiny negative solver errors within atol are clipped, then
    the result is normalized and checked. Matrix input is never modified.
    """
    atol = real_scalar(atol, "atol", 0)
    matrix = validate_stochastic(matrix)
    if not diagnose_chain(matrix).irreducible:
        raise ValueError("stationary solver requires an irreducible chain; reducible stationary distributions are not selected automatically")
    n = len(matrix)
    system = np.vstack((matrix.T-np.eye(n),np.ones(n)))
    target = np.concatenate((np.zeros(n),[1.]))
    pi,_,rank,singular_values = np.linalg.lstsq(system,target,rcond=None)
    if rank < n:
        raise ArithmeticError("stationary system is numerically rank deficient; inspect conditioning")
    condition = float(singular_values[0]/singular_values[-1])
    condition_limit = 1/np.sqrt(np.finfo(float).eps)
    if condition > condition_limit:
        raise ArithmeticError(
            f"stationary system is ill-conditioned (condition={condition:.3g} > "
            f"{condition_limit:.3g}); a small residual cannot establish an accurate "
            "stationary vector; use independent analysis or higher precision"
        )
    if not np.all(np.isfinite(pi)) or np.min(pi) < -atol:
        raise ArithmeticError("stationary solver returned invalid probabilities")
    pi = np.maximum(pi,0)
    if pi.sum() <= 0:
        raise ArithmeticError("stationary solver returned zero total probability")
    pi /= pi.sum()
    residual = float(np.max(np.abs(pi@matrix-pi)))
    if residual > atol or abs(pi.sum()-1) > atol:
        raise ArithmeticError(f"stationary residual {residual} exceeds tolerance {atol}")
    return pi
