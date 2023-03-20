:py:mod:`blackjax.mcmc.random_walk`
===================================

.. py:module:: blackjax.mcmc.random_walk

.. autoapi-nested-parse::

   Implements the (basic) user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels.
   Some interfaces are exposed here for convenience and for entry level users, who might be familiar
   with simpler versions of the algorithms, but in all cases they are particular instantiations
   of the Random Walk Rosenbluth-Metropolis-Hastings.

   Let's note x_{t-1} to the previous position and x_t to the newly sampled one.

   The variants offered are:

   1. Proposal distribution as addition of random noice from previous position. This means
   x_t = x_{t-1} + step. Function: `additive_step`

   2. Independent proposal distribution: P(x_t) doesn't depend on x_{t_1}. Function: `irmh`

   3. Proposal distribution using a symmetric function. That means P(x_t|x_{t-1}) = P(x_{t-1}|x_t).
    Function: `rmh` without proposal_logdensity_fn. See 'Metropolis Algorithm' in [1]

   4. Asymmetric proposal distribution. Function: `rmh` with proposal_logdensity_fn.
    See 'Metropolis-Hastings' Algorithm in [1]

   Reference: :cite:p:`gelman2014bayesian` Section 11.2

   .. rubric:: Examples

   The simplest case is:

   .. code::

       random_walk = blackjax.additive_step_random_walk(logdensity_fn, blackjax.mcmc.random_walk.normal(sigma))
       state = random_walk.init(position)
       new_state, info = random_walk.step(rng_key, state)

   In all cases we can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(random_walk.step)
       new_state, info = step(rng_key, state)



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.random_walk.RWState
   blackjax.mcmc.random_walk.RWInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.random_walk.normal
   blackjax.mcmc.random_walk.build_additive_step
   blackjax.mcmc.random_walk.build_irmh
   blackjax.mcmc.random_walk.build_rmh
   blackjax.mcmc.random_walk.build_rmh_transition_energy
   blackjax.mcmc.random_walk.rmh_proposal



.. py:function:: normal(sigma: blackjax.types.Array) -> Callable

   Normal Random Walk proposal.

   Propose a new position such that its distance to the current position is
   normally distributed. Suitable for continuous variables.

   Parameter
   ---------
   sigma:
       vector or matrix that contains the standard deviation of the centered
       normal distribution from which we draw the move proposals.



.. py:class:: RWState



   State of the RW chain.

   position
       Current position of the chain.
   log_density
       Current value of the log-density


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: logdensity
      :type: float

      


.. py:class:: RWInfo



   Additional information on the RW chain.

   This additional information can be used for debugging or computing
   diagnostics.

   acceptance_rate
       The acceptance probability of the transition, linked to the energy
       difference between the original and the proposed states.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.
   proposal
       The state proposed by the proposal.


   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: proposal
      :type: RWState

      


.. py:function:: build_additive_step()

   Build a Random Walk Rosenbluth-Metropolis-Hastings kernel

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: build_irmh() -> Callable

   Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
   that the proposal distribution does not depend on the particle being mutated :cite:p:`wang2022exact`.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: build_rmh()

   Build a Rosenbluth-Metropolis-Hastings kernel.
   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: build_rmh_transition_energy(proposal_logdensity_fn: Optional[Callable]) -> Callable


.. py:function:: rmh_proposal(logdensity_fn, transition_distribution, init_proposal, generate_proposal, sample_proposal: Callable = proposal.static_binomial_sampling) -> Callable


