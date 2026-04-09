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

.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> blackjax.mcmc.hmc.HMCState

   Create an initial state for the MHMCHMC kernel.

   :param position: Initial position of the chain.
   :param logdensity_fn: Log-density function of the target distribution.

   :rtype: The initial HMCState.


.. py:function:: build_kernel(integrator: Callable = integrators.isokinetic_mclachlan, divergence_threshold: float = 1000)

   Build an MHMCHMC kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the
                                transition is divergent.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: as_top_level_api(logdensity_fn: Callable, step_size: float, L_proposal_factor: float = jnp.inf, inverse_mass_matrix=1.0, *, divergence_threshold: int = 1000, integrator: Callable = integrators.isokinetic_mclachlan, num_integration_steps) -> blackjax.base.SamplingAlgorithm

   Implements the (basic) user interface for the MHMCHMC kernel.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param L_proposal_factor: Factor controlling partial momentum refreshment. ``jnp.inf`` disables
                             refreshment (standard HMC-like behavior).
   :param inverse_mass_matrix: Inverse mass matrix for the isokinetic integrator. Scalar or array.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent.
   :param integrator: The symplectic integrator to use to integrate the trajectory.
   :param num_integration_steps: Number of integration steps per transition.

   :rtype: A ``SamplingAlgorithm``.


