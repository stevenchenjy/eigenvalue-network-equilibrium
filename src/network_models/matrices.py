"""Transparent small-network constructors; no random walkers or hidden RNG.

All node numbers in code are zero-based. Networks derived from symmetric
weights are reversible, although their row-normalized matrices may not be
symmetric. Their stationary probabilities need not be uniform.
"""
import numpy as np

from .diagnostics import integer_parameter, real_array, real_scalar, validate_stochastic


def four_room(c):
    """Exact four-room model; endpoints c=0 and c=1 are diagnostic cases."""
    c = real_scalar(c, "c", 0, 1)
    return two_cluster(n_per_group=2, c=c)


def two_cluster(n_per_group=3, c=.1):
    """Choose the destination group with weights (1-c,c), then a room uniformly."""
    size = integer_parameter(n_per_group, "n_per_group", 1)
    c = real_scalar(c, "c", 0, 1)
    within = np.full((size,size), (1-c)/size)
    across = np.full((size,size), c/size)
    return validate_stochastic(np.block([[within, across], [across, within]]))


def normalize_weights(weights):
    """Normalize each nonnegative weighted-adjacency row; reject empty rows."""
    weights = real_array(weights, "weights")
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a nonempty square matrix")
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")
    with np.errstate(over="ignore"):
        totals = weights.sum(axis=1)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("weight rows must have positive finite sums")
    return validate_stochastic(weights / totals[:,None])


def uniform_network(n=4):
    """Every next node, including the current node, has probability 1/n."""
    n = integer_parameter(n, "n", 1)
    return validate_stochastic(np.full((n,n), 1/n))


def bottleneck_network(n_per_group=3, bridge_weight=.05, self_loop=1.):
    """Two complete groups with unit internal weights and one weighted bridge.

    Bridge joins node n_per_group-1 to n_per_group. Diagonal weights equal
    self_loop. Positive self-loops ensure aperiodicity for the default family.
    """
    size = integer_parameter(n_per_group, "n_per_group", 2)
    bridge_weight = real_scalar(bridge_weight, "bridge_weight", 0)
    self_loop = real_scalar(self_loop, "self_loop", 0)
    if bridge_weight == 0:
        raise ValueError("bridge_weight must be positive; use diagnostics for disconnected models")
    weights = np.zeros((2*size,2*size))
    weights[:size,:size] = 1
    weights[size:,size:] = 1
    np.fill_diagonal(weights, self_loop)
    weights[size-1,size] = weights[size,size-1] = bridge_weight
    return normalize_weights(weights)


def hub_spoke(n=5, laziness=.5):
    """A star centered at node 0, followed by P=laziness*I+(1-laziness)*Q.

    Leaves move to the hub under Q; the hub chooses a leaf uniformly. With
    laziness=0 the star is periodic; diagnostics, rather than this builder,
    decide whether a particular parameter is eligible for main experiments.
    """
    n = integer_parameter(n, "n", 2)
    laziness = real_scalar(laziness, "laziness", 0, 1)
    weights = np.zeros((n,n))
    weights[0,1:] = weights[1:,0] = 1
    return validate_stochastic(laziness*np.eye(n) + (1-laziness)*normalize_weights(weights))


def asymmetric_network(n=5, laziness=.5, bias=.8):
    """Optional lazy directed ring: bias clockwise, 1-bias anticlockwise.

    This circulant example can have complex eigenvalues. It does not represent
    every nonreversible chain or establish a general claim about nonnormality.
    """
    n = integer_parameter(n, "n", 3)
    laziness = real_scalar(laziness, "laziness", 0, 1)
    bias = real_scalar(bias, "bias", 0, 1)
    matrix = np.eye(n)*laziness
    for node in range(n):
        matrix[node,(node+1)%n] += (1-laziness)*bias
        matrix[node,(node-1)%n] += (1-laziness)*(1-bias)
    return validate_stochastic(matrix)
