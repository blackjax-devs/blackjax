"""
strategies to tune the parameters of mcmc kernels
used within smc, based on MCMC states
"""
import jax.numpy as jnp

__all__ = ["update_scale_from_acceptance_rate"]

TARGET_MH_ACCEPTANCE_RATE = 0.234


def update_scale_from_acceptance_rate(scales, acceptance_rates):
    """
    Given N chains from some MCMC algorithm like Random Walk Metropolis
    and N scale factors, each associated to a different chain.
    Updates the scale factors taking into account acceptance rates and
    the average acceptance rate.

    Under certain assumptions it is known that the optimal acceptance rate
    of Metropolis Hastings is 0.4 for 1 dimension and converges to
    0.234 in infinite dimensions. In practice, 0.234 is a reasonable
    assumption for 5 or more dimensions.

    If certain chain is below optimal acceptance rate, its scale will decrease
    and if its above, its scale will increase,
    -------
    """
    chain_scales = jnp.exp(
        jnp.log(scales) + acceptance_rates - TARGET_MH_ACCEPTANCE_RATE
    )
    return 0.5 * (chain_scales + chain_scales.mean())
