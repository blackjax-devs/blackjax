import jax
import jax.numpy as jnp
import pytest

from blackjax.mcmc.slingshot import SlingshotInfo, SlingshotState, build_kernel, init


def test_callable_step_size_initialization():
    rng_key = jax.random.PRNGKey(0)
    position = jnp.array([1.0, 2.0])

    def logdensity_fn(x):
        return -jnp.sum(x**2)

    def callable_step_size(state):
        return 0.5

    # Initialize the state
    state = init(position, logdensity_fn)

    # Build the kernel with a callable step_size
    kernel = build_kernel(logdensity_fn, step_size=callable_step_size)

    # Verify that the kernel can be called without errors
    try:
        next_state, info = kernel(state, rng_key)
    except Exception as e:
        pytest.fail(f"Kernel initialization with callable step_size failed: {e}")

    # Further assertions can be added here if needed, e.g., to check the type of next_state or info
    assert isinstance(next_state, SlingshotState)
    assert isinstance(info, SlingshotInfo)
    assert next_state.position.shape == position.shape
    assert info.proposal_cloud.shape == (
        1000,
        position.shape[0],
    )  # Default num_proposals is 1000

    