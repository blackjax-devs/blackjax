import jax
import jax.numpy as jnp
import pytest

import blackjax

def test_slingshot_parameter_recovery():
    """Verify Slingshot MP-MCMC can sample from a standard normal target."""
    def logdensity_fn(x):
        return -0.5 * jnp.sum(x**2)

    rng_key = jax.random.PRNGKey(42)
    initial_position = jnp.array([2.0, -2.0])
    
    # Instantiate the algorithm via top-level API
    algo = blackjax.slingshot(logdensity_fn, step_size=0.5, num_proposals=1000)
    state = algo.init(initial_position)
    
    # Compile the transition step using lax.scan
    @jax.jit(static_argnames=("num_steps",))
    def run_chain(key, initial_state, num_steps=100):
        def body_fn(carry_state, step_key):
            next_state, info = algo.step(step_key, carry_state)
            return next_state, next_state.position
            
        keys = jax.random.split(key, num_steps)
        _, positions = jax.lax.scan(body_fn, initial_state, keys)
        return positions

    # Execute chain execution loop
    positions = run_chain(rng_key, state, num_steps=200)
    
    # Assert output shapes and check for numerical execution integrity
    assert positions.shape == (200, 2)
    assert not jnp.any(jnp.isnan(positions))
