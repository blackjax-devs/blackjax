import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from blackjax.mcmc.random_walk import (
    RWState,
    build_additive_step,
    build_irmh,
    build_rmh_transition_energy,
    rmh_proposal,
)


class AdditiveStepTest(unittest.TestCase):
    def test_one_step_addition(self):
        """New position is an addition to previous position.
        Since the density == 1, the proposal is accepted.
        The random step may depend on the previous position
        """
        rng_key = jax.random.key(0)
        initial_position = jnp.array([50.0])

        def random_step(key, position):
            assert position == initial_position
            return jnp.array([10.0])

        def test_logdensity_accepts(position):
            """
            a logdensity that ges maximized after the step
            """
            return 0.0 if all(position > 59.0) else 0.5

        step = build_additive_step()

        new_state, _ = step(
            rng_key,
            RWState(position=initial_position, logdensity=1.0),
            test_logdensity_accepts,
            random_step,
        )

        np.testing.assert_allclose(new_state.position, jnp.array([60.0]))

        assert new_state.position


class IRMHTest(unittest.TestCase):
    def proposal_distribution(self, key):
        return jnp.array([10.0])

    def logdensity_accepts(self, position):
        """
        a logdensity that gets maximized after the step
        """
        return 0.0 if all(position - 10.0 < 1e-10) else 0.5

    def test_proposal_is_independent_of_position(self):
        """New position does not depend on previous position"""
        rng_key = jax.random.key(0)
        initial_position = jnp.array([50.0])
        other_position = jnp.array([15000.0])

        step = build_irmh()

        for previous_position in [initial_position, other_position]:
            new_state, state_info = step(
                rng_key,
                RWState(position=previous_position, logdensity=1.0),
                self.logdensity_accepts,
                self.proposal_distribution,
            )
            np.testing.assert_allclose(new_state.position, jnp.array([10.0]))
            np.testing.assert_allclose(state_info.acceptance_rate, 0.367879, rtol=1e-5)

    def test_non_symmetric_proposal(self):
        """
        Given that proposal_logdensity_fn is included,
        thus the proposal is non-symmetric.
        When computing the acceptance of the proposed state
        Then proposal_logdensity_fn value is taken into account
        """
        rng_key = jax.random.key(0)
        initial_position = jnp.array([50.0])

        def test_proposal_logdensity(new_state, prev_state):
            return 0.1 if all(new_state.position - 10 < 1e-10) else 0.5

        step = build_irmh()

        for previous_position in [initial_position]:
            _, state_info = step(
                rng_key,
                RWState(position=previous_position, logdensity=1.0),
                self.logdensity_accepts,
                self.proposal_distribution,
                test_proposal_logdensity,
            )

            np.testing.assert_allclose(state_info.acceptance_rate, 0.246597)


class RMHProposalTest(unittest.TestCase):
    def transition_distribution(self, key, position):
        return jnp.array([10.0])

    def reject(self, key, log_p_accept, previous_proposal, new_proposal):
        return previous_proposal, (False, 0.3, None)

    def accept(self, key, log_p_accept, previous_proposal, new_proposal):
        return new_proposal, (True, 0.5, None)

    def compute_ratio(self, new_state, prev_state):
        return 0.5

    def test_generate_reject(self):
        """
        Steps from previous state,
        Builds a proposal from the new state
        and given that the sampling rule rejects,
        the prev_state is proposed again
        """
        rng_key = jax.random.key(0)

        prev_state = RWState(jnp.array([30.0]), 15.0)

        generate = rmh_proposal(
            logdensity_fn=lambda _: 50.0,
            transition_distribution=self.transition_distribution,
            compute_acceptance_ratio=self.compute_ratio,
            sample_proposal=self.reject,
        )

        sampled_proposal, do_accept, p_accept = generate(rng_key, prev_state)

        assert not do_accept
        assert p_accept == 0.3
        np.testing.assert_allclose(sampled_proposal.position, jnp.array([30.0]))

    def test_generate_accept(self):
        rng_key = jax.random.key(0)
        prev_state = RWState(jnp.array([30.0]), 15.0)

        generate = rmh_proposal(
            logdensity_fn=lambda _: 50.0,
            transition_distribution=self.transition_distribution,
            compute_acceptance_ratio=self.compute_ratio,
            sample_proposal=self.accept,
        )
        sampled_proposal, do_accept, p_accept = generate(rng_key, prev_state)

        np.testing.assert_allclose(sampled_proposal.position, jnp.array([10.0]))


class RMHTransitionEnergyTest(unittest.TestCase):
    def test_energy(self):
        one_state = RWState(None, 50.0)
        another_state = RWState(None, 30.0)

        def proposal_logdensity(new_state, prev_state):
            return 100 if new_state == one_state else 200

        energy = build_rmh_transition_energy(None)
        np.testing.assert_allclose(energy(one_state, another_state), -30.0)
        np.testing.assert_allclose(energy(another_state, one_state), -50.0)

        energy = build_rmh_transition_energy(proposal_logdensity)
        np.testing.assert_allclose(energy(one_state, another_state), -230)
        np.testing.assert_allclose(energy(another_state, one_state), -150)


if __name__ == "__main__":
    absltest.main()
