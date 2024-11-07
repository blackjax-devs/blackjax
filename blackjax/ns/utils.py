import jax
import jax.numpy as jnp

from blackjax.ns.base_ns import NSInfo


def compute_nlive(info: NSInfo):
    """Compute number of live points from birth and death contours.

    Parameters
    ----------
    death, birth : array-like
        list of birth and death contours

    Returns
    -------
    nlive : jnp.array
        number of live points at each contour
    """
    birth = info.logL_birth
    death = info.logL

    # Combine birth and death arrays
    combined = jnp.concatenate(
        [
            jnp.column_stack((birth, jnp.ones_like(birth))),
            jnp.column_stack((death, -jnp.ones_like(death))),
        ]
    )

    # Sort the combined array
    sorted_indices = jnp.lexsort((combined[:, 1], combined[:, 0]))
    sorted_combined = combined[sorted_indices]

    # Compute cumulative sum
    cumsum = jnp.cumsum(sorted_combined[:, 1])

    # Extract the values corresponding to death contours
    death_mask = sorted_combined[:, 1] == -1
    nlive = cumsum[death_mask] + 1

    return nlive


def logX(key: jax.random.PRNGKey, dead: NSInfo, samples=100):
    """Log-Volume.

    The log of the prior volume contained within each iso-likelihood
    contour.

    Parameters
    ----------
    nsamples : int, optional
        - If nsamples is not supplied, calculate mean value
        - If nsamples is integer, draw nsamples from the distribution of
          values inferred by nested sampling

    Returns
    -------
    if nsamples is None:
        WeightedSeries like self
    elif nsamples is int:
        WeightedDataFrame like self, columns range(nsamples)
    """
    key, subkey = jax.random.split(key)
    r = jnp.log(
        jax.random.uniform(subkey, shape=(dead.logL.shape[0], samples), minval=1e-6)
    ) #TODO Don't hardcode minval as 1e-6, read the precision appropriately
    nlive = compute_nlive(dead)
    t = r / nlive[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)
    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate([logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0)
    logdX = jnp.log(1 - jnp.exp(logXm - logXp)) + logXp - jnp.log(2)
    return logX, logdX


def log_weights(key: jax.random.PRNGKey, dead: NSInfo, samples=100, beta=1.0):
    _, ldX = logX(key, dead, samples)
    return ldX + beta * dead.logL[..., jnp.newaxis]
