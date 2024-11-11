import jax
import jax.numpy as jnp

from blackjax.ns.base import NSInfo


def compute_nlive(info: NSInfo):
    """
    Compute the effective number of live points at each death contour.

    Parameters
    ----------
    info : NSInfo
        Contains log-likelihood arrays for birth and death contours.

    Returns
    -------
    nlive : jnp.array
        Number of live points at each contour.
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
    sorted_indices = jnp.lexsort((combined[:, 1], combined[:, 0]))
    sorted_combined = combined[sorted_indices]
    cumsum = jnp.cumsum(sorted_combined[:, 1])

    death_mask = sorted_combined[:, 1] == -1
    nlive = cumsum[death_mask] + 1

    return nlive


def logX(key: jax.random.PRNGKey, dead: NSInfo, samples=100):
    """Compute the log of the prior volume within iso-likelihood contours.

    This function calculates the log volume of the prior contained within
    each iso-likelihood contour for a nested sampling result.

    Parameters
    ----------
    key : jax.random.PRNGKey
        a jax rng key.
    dead : NSInfo
        An object containing the log-likelihoods and other relevant
        information of the dead points in nested sampling.
    samples : int, optional
        The number of samples to draw. Default is 100.

    Returns
    -------
    logX : jnp.ndarray
        Cumulative log volume for each sample.
    logdX : jnp.ndarray
        Logarithm of the difference in volume for each contour.
    """
    key, subkey = jax.random.split(key)
    min_val = jnp.finfo(dead.logL.dtype).tiny
    r = jnp.log(
        jax.random.uniform(subkey, shape=(dead.logL.shape[0], samples)).clip(
            min_val, 1 - min_val
        )
    )

    nlive = compute_nlive(dead)
    t = r / nlive[:, jnp.newaxis]
    logX = jnp.cumsum(t, axis=0)

    logXp = jnp.concatenate([jnp.zeros((1, logX.shape[1])), logX[:-1]], axis=0)
    logXm = jnp.concatenate(
        [logX[1:], jnp.full((1, logX.shape[1]), -jnp.inf)], axis=0
    )
    log_diff = logXm - logXp
    logdX = jnp.log1p(-jnp.exp(log_diff).clip(max=1.0)) + logXp - jnp.log(2)
    return logX, logdX


def log_weights(key: jax.random.PRNGKey, dead: NSInfo, samples=100, beta=1.0):
    """
    Calculate the log importance weights for Nested Sampling results.

    Parameters
    ----------
    key : jax.random.PRNGKey
        a jax rng key.
    dead : NSInfo
        An object containing the log-likelihoods and other relevant
        information of the dead points in nested sampling.
    samples : int, optional
        The number of samples to draw for estimating log weights, by default 100.
    beta : float, optional
        The inverse temperature of the log-likelihood to calculate at,
        by default 1.0.

    Returns
    -------
    jnp.ndarray
        An array containing the log weights of the dead points.
    """
    _, ldX = logX(key, dead, samples)
    return ldX + beta * dead.logL[..., jnp.newaxis]
