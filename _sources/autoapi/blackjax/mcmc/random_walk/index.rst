:py:mod:`blackjax.mcmc.random_walk`
===================================

.. py:module:: blackjax.mcmc.random_walk

.. autoapi-nested-parse::

   Implements the (basic) user interfaces for Random Walk Rosenbluth-Metropolis-Hastings kernels.
   Some interfaces are exposed here for convenience and for entry level users, who might be familiar
   with simpler versions of the algorithms, but in all cases they are particular instantiations
   of the Random Walk Rosenbluth-Metropolis-Hastings.

   Let's note $x_{t-1}$ to the previous position and $x_t$ to the newly sampled one.

   The variants offered are:

   1. Proposal distribution as addition of random noice from previous position. This means
      $x_t = x_{t-1} + step$.

       Function: `additive_step`

   2. Independent proposal distribution: $P(x_t)$ doesn't depend on $x_{t_1}$.

       Function: `irmh`

   3. Proposal distribution using a symmetric function. That means $P(x_t|x_{t-1}) = P(x_{t-1}|x_t)$.
      See 'Metropolis Algorithm' in [1].

       Function: `rmh` without proposal_logdensity_fn.

   4. Asymmetric proposal distribution. See 'Metropolis-Hastings' Algorithm in [1].

       Function: `rmh` with proposal_logdensity_fn.

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
   blackjax.mcmc.random_walk.additive_step_random_walk
   blackjax.mcmc.random_walk.irmh
   blackjax.mcmc.random_walk.rmh



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
      :type: blackjax.types.ArrayTree

      

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


.. py:class:: additive_step_random_walk


   Implements the user interface for the Additive Step RMH

   .. rubric:: Examples

   A new kernel can be initialized and used with the following code:

   .. code::

       rw = blackjax.additive_step_random_walk(logdensity_fn, random_step)
       state = rw.init(position)
       new_state, info = rw.step(rng_key, state)

   The specific case of a Gaussian `random_step` is already implemented, either with independent components
   when `covariance_matrix` is a one dimensional array or with dependent components if a two dimensional array:

   .. code::

       rw_gaussian = blackjax.additive_step_random_walk.normal_random_walk(logdensity_fn, covariance_matrix)
       state = rw_gaussian.init(position)
       new_state, info = rw_gaussian.step(rng_key, state)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param random_step: A Callable that takes a random number generator and the current state and produces a step,
                       which will be added to the current position to obtain a new position. Must be symmetric
                       to maintain detailed balance. This means that P(step|position) = P(-step | position+step)

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      

   .. py:method:: normal_random_walk(logdensity_fn: Callable, sigma)
      :classmethod:

      :param logdensity_fn: The log density probability density function from which we wish to sample.
      :param sigma: The value of the covariance matrix of the gaussian proposal distribution.

      :rtype: A ``SamplingAlgorithm``.



.. py:function:: build_irmh() -> Callable

   Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
   that the proposal distribution does not depend on the particle being mutated :cite:p:`wang2022exact`.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: irmh


   Implements the (basic) user interface for the independent RMH.

   .. rubric:: Examples

   A new kernel can be initialized and used with the following code:

   .. code::

       rmh = blackjax.irmh(logdensity_fn, proposal_distribution)
       state = rmh.init(position)
       new_state, info = rmh.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(rmh.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param proposal_distribution: A Callable that takes a random number generator and produces a new proposal. The
                                 proposal is independent of the sampler's current state.
   :param proposal_logdensity_fn: For non-symmetric proposals, a function that returns the log-density
                                  to obtain a given proposal knowing the current state. If it is not
                                  provided we assume the proposal is symmetric.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


.. py:function:: build_rmh()

   Build a Rosenbluth-Metropolis-Hastings kernel.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: rmh


   Implements the user interface for the RMH.

   .. rubric:: Examples

   A new kernel can be initialized and used with the following code:

   .. code::

       rmh = blackjax.rmh(logdensity_fn, proposal_generator)
       state = rmh.init(position)
       new_state, info = rmh.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(rmh.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param proposal_generator: A Callable that takes a random number generator and the current state and produces a new proposal.
   :param proposal_logdensity_fn:
                                  The logdensity function associated to the proposal_generator. If the generator is non-symmetric,
                                   P(x_t|x_t-1) is not equal to P(x_t-1|x_t), then this parameter must be not None in order to apply
                                   the Metropolis-Hastings correction for detailed balance.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


.. py:function:: build_rmh_transition_energy(proposal_logdensity_fn: Optional[Callable]) -> Callable


.. py:function:: rmh_proposal(logdensity_fn: Callable, transition_distribution: Callable, compute_acceptance_ratio: Callable, sample_proposal: Callable = proposal.static_binomial_sampling) -> Callable


