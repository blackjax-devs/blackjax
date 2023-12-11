:py:mod:`blackjax.mcmc.dynamic_hmc`
===================================

.. py:module:: blackjax.mcmc.dynamic_hmc

.. autoapi-nested-parse::

   Public API for the Dynamic HMC Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.dynamic_hmc.DynamicHMCState
   blackjax.mcmc.dynamic_hmc.dynamic_hmc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.dynamic_hmc.init
   blackjax.mcmc.dynamic_hmc.build_kernel
   blackjax.mcmc.dynamic_hmc.halton_sequence



.. py:class:: DynamicHMCState




   State of the dynamic HMC algorithm.

   Adds a utility array for generating a pseudo or quasi-random sequence of
   number of integration steps.


   .. py:attribute:: position
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: random_generator_arg
      :type: blackjax.types.Array

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable, random_generator_arg: blackjax.types.Array)


.. py:function:: build_kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000, next_random_arg_fn: Callable = lambda key: jax.random.split(key)[1], integration_steps_fn: Callable = lambda key: jax.random.randint(key, (), 1, 10))

   Build a Dynamic HMC kernel where the number of integration steps is chosen randomly.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the transition is divergent.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param integration_steps_fn: Function that generates the next pseudo or quasi-random number of integration steps in the
                                sequence, given the current `random_generator_arg`. Needs to return an `int`.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: dynamic_hmc


   Implements the (basic) user interface for the dynamic HMC kernel.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the trajectory.
   :param next_random_arg_fn: Function that generates the next `random_generator_arg` from its previous value.
   :param integration_steps_fn: Function that generates the next pseudo or quasi-random number of integration steps in the
                                sequence, given the current `random_generator_arg`.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


.. py:function:: halton_sequence(i: blackjax.types.Array, max_bits: int = 10) -> float


