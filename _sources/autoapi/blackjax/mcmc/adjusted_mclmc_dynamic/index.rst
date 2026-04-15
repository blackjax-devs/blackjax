blackjax.mcmc.adjusted_mclmc_dynamic
====================================

.. py:module:: blackjax.mcmc.adjusted_mclmc_dynamic

.. autoapi-nested-parse::

   Public API for the Metropolis Hastings Microcanonical Hamiltonian Monte Carlo (MHMCHMC) Kernel. This is closely related to the Microcanonical Langevin Monte Carlo (MCLMC) Kernel, which is an unadjusted method. This kernel adds a Metropolis-Hastings correction to the MCLMC kernel. It also only refreshes the momentum variable after each MH step, rather than during the integration of the trajectory. Hence "Hamiltonian" and not "Langevin".



Functions
---------

.. autoapisummary::

   blackjax.mcmc.adjusted_mclmc_dynamic.init
   blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel
   blackjax.mcmc.adjusted_mclmc_dynamic.as_top_level_api


Module Contents
---------------

.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: blackjax.types.Array) -> blackjax.mcmc.dynamic_hmc.DynamicHMCState

   Create an initial state for the dynamic MHMCHMC kernel.

   :param position: Initial position of the chain.
   :param logdensity_fn: Log-density function of the target distribution.
   :param random_generator_arg: Argument passed to ``integration_steps_fn`` and ``next_random_arg_fn``
                                to generate the number of integration steps.

   :rtype: The initial DynamicHMCState.


.. py:function:: build_kernel(integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10), integrator: Callable = integrators.isokinetic_mclachlan, divergence_threshold: float = 1000, next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1])

   Build a Dynamic MHMCHMC kernel where the number of integration steps is chosen randomly.

   :param integration_steps_fn: Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
                                that draws the number of integration steps for a single transition.
                                Extra positional arguments beyond ``random_generator_arg`` are supplied
                                at call time via ``integration_steps_params`` on the inner kernel, so
                                tunable parameters (e.g. average number of steps, distribution bounds)
                                can be adapted without rebuilding the kernel.
   :param integrator: The integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the transition is divergent.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, step_size: float, L_proposal_factor: float = jnp.inf, inverse_mass_matrix=1.0, *, divergence_threshold: int = 1000, integrator: Callable = integrators.isokinetic_mclachlan, next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1], integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10), integration_steps_params: tuple = ()) -> blackjax.base.SamplingAlgorithm

   Implements the (basic) user interface for the dynamic MHMCHMC kernel.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param integration_steps_fn: Callable with signature ``(random_generator_arg, *integration_steps_params) -> int``
                                that draws the number of integration steps for a single transition.
   :param integration_steps_params: Extra positional arguments unpacked into ``integration_steps_fn`` after
                                    ``random_generator_arg`` on every step.  Use this to pass tunable
                                    parameters (e.g. ``(avg_num_integration_steps,)`` or
                                    ``(lower_bound, upper_bound)``) without rebuilding the kernel.
                                    Defaults to ``()`` so that a plain 1-arg ``integration_steps_fn`` works
                                    unchanged.

   :rtype: A ``SamplingAlgorithm``.


