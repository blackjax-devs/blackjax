import jax
import numpy as np
from jax import numpy as jnp

from blackjax.types import PyTree


class MultivariableParticlesGenerator:
    """
    Builds particles for tests belonging to a posterior with more than one variable. Let's assume we want to
    sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
    """

    def __init__(self, n_particles, mean_x=None, mean_y=None, cov_x=None, cov_y=None):
        self.n_particles = n_particles
        self.mean_x = mean_x if mean_x is not None else [10.0, 5.0]
        self.mean_y = mean_y if mean_y is not None else [0.0, 0.0]
        self.cov_x = cov_x if cov_x is not None else [[1.0, 0.0], [0.0, 1.0]]
        self.cov_y = cov_y if cov_y is not None else [[1.0, 0.0], [0.0, 1.0]]

    def particles(self):
        return [
            np.random.multivariate_normal(
                mean=self.mean_x, cov=self.cov_x, size=self.n_particles
            ),
            np.random.multivariate_normal(
                mean=self.mean_y, cov=self.cov_y, size=self.n_particles
            ),
        ]


def particles_from_multivariable_posterior(n_particles):
    """
    Builds particles belonging to a posterior with more than one variable. Let's assume we want to
    sample from P(x,y) x ~ N(mean, cov) y ~ N(mean, cov)
    """
    return [
        np.random.multivariate_normal(
            mean=[10.0, 5.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=n_particles
        ),
        np.random.multivariate_normal(
            mean=[3.0, 20.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=n_particles
        ),
    ]


def logprob_fn_for_multivariable_posterior(position: PyTree):
    def single_particle_logprobfn(particle):
        return jax.scipy.stats.multivariate_normal.logpdf(
            particle, mean=np.array([0.0, 0.0]), cov=jnp.diag(jnp.ones(2))
        )

    return sum(jax.tree_map(single_particle_logprobfn, position))


def log_weights_fn_for_multivariable_posterior(x, y=1):
    return jnp.sum(jax.scipy.stats.norm.logpdf(y - x))
