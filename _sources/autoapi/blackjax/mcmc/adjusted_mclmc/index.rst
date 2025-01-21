blackjax.mcmc.adjusted_mclmc
============================

.. py:module:: blackjax.mcmc.adjusted_mclmc

.. autoapi-nested-parse::

   Public API for the Metropolis Hastings Microcanonical Hamiltonian Monte Carlo (MHMCHMC) Kernel. This is closely related to the Microcanonical Langevin Monte Carlo (MCLMC) Kernel, which is an unadjusted method. This kernel adds a Metropolis-Hastings correction to the MCLMC kernel. It also only refreshes the momentum variable after each MH step, rather than during the integration of the trajectory. Hence "Hamiltonian" and not "Langevin".

   NOTE: For best performance, we recommend using adjusted_mclmc_dynamic instead of this module, which is primarily intended for use in parallelized versions of the algorithm.



Functions
---------

.. autoapisummary::

   blackjax.mcmc.adjusted_mclmc.init
   blackjax.mcmc.adjusted_mclmc.build_kernel
   blackjax.mcmc.adjusted_mclmc.as_top_level_api


Module Contents
---------------

.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable)

.. py:function:: build_kernel(logdensity_fn: Callable, integrator: Callable = integrators.isokinetic_mclachlan, divergence_threshold: float = 1000, inverse_mass_matrix=1.0)

   Build an MHMCHMC kernel where the number of integration steps is chosen randomly.

   :param integrator: The integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the transition is divergent.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param integration_steps_fn: Function that generates the next pseudo or quasi-random number of integration steps in the
                                sequence, given the current `random_generator_arg`. Needs to return an `int`.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, step_size: float, L_proposal_factor: float = jnp.inf, inverse_mass_matrix=1.0, *, divergence_threshold: int = 1000, integrator: Callable = integrators.isokinetic_mclachlan, num_integration_steps) -> blackjax.base.SamplingAlgorithm

   Implements the (basic) user interface for the MHMCHMC kernel.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param integration_steps_fn: Function that generates the next pseudo or quasi-random number of integration steps in the
                                sequence, given the current `random_generator_arg`.

   :rtype: A ``SamplingAlgorithm``.


