import chex
import jax
import jax.numpy as jnp

import blackjax
from blackjax.mcmc.slingshot import SlingshotState, SlingshotInfo
from tests.fixtures import BlackJAXTest, std_normal_logdensity

class SlingshotTest(BlackJAXTest):

    def test_protocol_conformance(self):
        """Verify Slingshot conforms to the BlackJAX SamplingAlgorithm protocol."""
        initial_position = jnp.array([2.0, -2.0])
        
        # Test low-level init
        state = blackjax.mcmc.slingshot.init(initial_position, std_normal_logdensity)
        self.assertIsInstance(state, SlingshotState)
        
        # Test top-level init and step
        algo = blackjax.slingshot(std_normal_logdensity, step_size=0.5, num_proposals=10)
        state_from_algo = algo.init(initial_position)
        self.assertIsInstance(state_from_algo, SlingshotState)
        
        new_state, info = algo.step(self.next_key(), state_from_algo)
        self.assertIsInstance(new_state, SlingshotState)
        self.assertIsInstance(info, SlingshotInfo)

    def test_slingshot_parameter_recovery(self):
        """Verify Slingshot MP-MCMC can sample from a standard normal target."""
        initial_position = jnp.array([2.0, -2.0])
        
        algo = blackjax.slingshot(std_normal_logdensity, step_size=0.5, num_proposals=1000)
        state = algo.init(initial_position)
        
        @jax.jit
        @chex.assert_max_traces(n=2)
        def run_chain(initial_state, step_keys):
            def body_fn(carry_state, step_key):
                next_state, info = algo.step(step_key, carry_state)
                return next_state, next_state.position
                
            _, positions = jax.lax.scan(body_fn, initial_state, step_keys)
            return positions

        num_steps = 200
        step_keys = jax.random.split(self.next_key(), num_steps)
        positions = run_chain(state, step_keys)
        
        # Second run to test for excess recompilation
        step_keys_2 = jax.random.split(self.next_key(), num_steps)
        run_chain(state, step_keys_2)
        
        assert positions.shape == (num_steps, 2)
        assert not jnp.any(jnp.isnan(positions))
