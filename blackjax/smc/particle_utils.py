"""
In the SMC codebase, particles are represented using
[V1, ..., VN] where each V is an array associated
with a random variable from a distribution. If the ith-variable
belongs to some (possibly distinct) R^{n} subspace R_i. Each V has shape
(n_particles, |R_i|).
If N=1 then no list is used, and particles are represented with an array.
"""

import jax

from blackjax.types import PyTree


def number_of_particles(particles: PyTree) -> int:
    return jax.tree_flatten(particles)[0][0].shape[0]


def posterior_variables(particles: PyTree) -> int:
    if isinstance(particles, list):
        return len(particles)
    else:
        return 1
