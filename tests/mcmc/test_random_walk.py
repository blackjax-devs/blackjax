import functools
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from blackjax.mcmc.proposal import Proposal
from blackjax.mcmc.random_walk import (
    RWState,
    build_additive_step,
    build_irmh,
    build_rmh_transition_energy,
    rmh_proposal,
)


class AdditiveStepTest(chex.TestCase):
    @chex.all_variants
    def test_one_step_addition(self):
        """New position is an addition to previous position.
        Since the density == 1, the proposal is accepted.
        The random step may depend on the previous position
        """
        rng_key = jax.random.PRNGKey(0)
        initial_position = jnp.array([50.0])

        @chex.chexify
        def random_step(key, position):
            chex.assert_tree_all_close(position, initial_position)
            return jnp.array([10.0])

        def test_logdensity_accepts(position):
            """
            a logdensity that ges maximized after the step
            """
            return jnp.where(position > 59.0, 0.0, 0.5).sum()

        step = build_additive_step()
        wrapped = self.variant(
            partial(
                step, random_step=random_step, logdensity_fn=test_logdensity_accepts
            )
        )
        new_state, _ = wrapped(
            rng_key, state=RWState(position=initial_position, logdensity=1.0)
        )

        np.testing.assert_allclose(new_state.position, jnp.array([60.0]))

        assert new_state.position


class IRMHTest(chex.TestCase):
    @chex.all_variants(with_jit=True)
    def test_proposal_is_independent_of_position(self):
        """New position does not depend on previous"""
        rng_key = jax.random.PRNGKey(0)
        initial_position = jnp.array([50.0])
        other_position = jnp.array([15000.0])

        def proposal_distribution(key):
            return jnp.array([10.0])

        def test_logdensity_accepts(position):
            """
            a logdensity that gets maximized after the step
            """
            return jnp.where(position - 10.0 < 1e-10, 0.0, 0.5).sum()

        step = build_irmh()
        wrapped = self.variant(
            functools.partial(
                step,
                proposal_distribution=proposal_distribution,
                logdensity_fn=test_logdensity_accepts,
            )
        )

        for previous_position in [initial_position, other_position]:
            new_state, _ = wrapped(
                rng_key, state=RWState(position=previous_position, logdensity=1.0)
            )
            np.testing.assert_allclose(new_state.position, jnp.array([10.0]))


class RMHProposalTest(chex.TestCase):
    def transition_distribution(self, key, position):
        return jnp.array([10.0])

    def reject(self, key, previous_proposal, new_proposal):
        return previous_proposal, False, 0.3

    def accept(self, key, previous_proposal, new_proposal):
        return new_proposal, True, 0.5

    def init_proposal(self, state):
        return Proposal(state, 0, 0, 0)

    def generate_proposal(self, prev, new):
        return Proposal(new, 0, 0, 0), False

    @chex.all_variants(with_jit=True)
    def test_generate_reject(self):
        """
        Steps from previous state,
        Builds a proposal from the new state
        and given that the sampling rule rejects,
        the prev_state is proposed again
        """
        rng_key = jax.random.PRNGKey(0)

        prev_state = RWState(jnp.array([30.0]), 15.0)

        generate = rmh_proposal(
            logdensity_fn=lambda position: 50.0,
            transition_distribution=self.transition_distribution,
            init_proposal=self.init_proposal,
            generate_proposal=self.generate_proposal,
            sample_proposal=self.reject,
        )

        sampled_proposal, do_accept, p_accept = self.variant(generate)(
            rng_key, prev_state
        )

        assert not do_accept
        assert p_accept == 0.3
        np.testing.assert_allclose(sampled_proposal.state.position, jnp.array([30.0]))

    def test_generate_accept(self):
        rng_key = jax.random.PRNGKey(0)
        prev_state = RWState(jnp.array([30.0]), 15.0)

        generate = rmh_proposal(
            logdensity_fn=lambda position: 50.0,
            transition_distribution=self.transition_distribution,
            init_proposal=self.init_proposal,
            generate_proposal=self.generate_proposal,
            sample_proposal=self.accept,
        )
        sampled_proposal, do_accept, p_accept = generate(rng_key, prev_state)

        np.testing.assert_allclose(sampled_proposal.state.position, jnp.array([10.0]))


class RMHTransitionEnergyTest(chex.TestCase):
    @chex.all_variants(with_jit=True)
    def test_energy(self):
        one_state = RWState(None, 50.0)
        another_state = RWState(None, 30.0)

        def proposal_logdensity(new_state, prev_state):
            return jnp.where(
                jnp.abs(new_state.logdensity - 50.0) < 1e-10, 100, 200
            ).sum()

        energy = self.variant(build_rmh_transition_energy(None))
        np.testing.assert_allclose(energy(one_state, another_state), -30.0)
        np.testing.assert_allclose(energy(another_state, one_state), -50.0)

        energy = self.variant(build_rmh_transition_energy(proposal_logdensity))
        np.testing.assert_allclose(energy(one_state, another_state), -230)
        np.testing.assert_allclose(energy(another_state, one_state), -150)


if __name__ == "__main__":
    absltest.main()
