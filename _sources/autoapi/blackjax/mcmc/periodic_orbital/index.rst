:py:mod:`blackjax.mcmc.periodic_orbital`
========================================

.. py:module:: blackjax.mcmc.periodic_orbital

.. autoapi-nested-parse::

   Public API for Periodic Orbital Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.periodic_orbital.PeriodicOrbitalState
   blackjax.mcmc.periodic_orbital.orbital_hmc



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.periodic_orbital.init
   blackjax.mcmc.periodic_orbital.build_kernel



.. py:class:: PeriodicOrbitalState




   State of the periodic orbital algorithm.

   The periodic orbital algorithm takes one orbit with weights,
   samples from the points on that orbit according to their weights
   and returns another weighted orbit of the same period.

   positions
       a collection of points on the orbit, representing samples from
       the target distribution.
   weights
       weights of each point on the orbit, reweights points to ensure
       they are from the target distribution.
   directions
       an integer indicating the position on the orbit of each point.
   logdensities
       vector with logdensities (negative potential energies) for each point in
       the orbit.
   logdensities_grad
       matrix where each row is a vector with gradients of the logdensity
       function for each point in the orbit.

   .. py:attribute:: positions
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: weights
      :type: blackjax.types.Array

      

   .. py:attribute:: directions
      :type: blackjax.types.Array

      

   .. py:attribute:: logdensities
      :type: blackjax.types.Array

      

   .. py:attribute:: logdensities_grad
      :type: blackjax.types.ArrayTree

      


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable, period: int) -> PeriodicOrbitalState

   Create a periodic orbital state from a position.

   :param position: the current values of the random variables whose posterior we want to
                    sample from. Can be anything from a list, a (named) tuple or a dict of
                    arrays. The arrays can either be Numpy or JAX arrays.
   :param logdensity_fn: a function that returns the value of the log posterior when called
                         with a position.
   :param period: the number of steps used to build the orbit

   :returns: * A periodic orbital state that repeats the same position for `period` times,
             * *sets equal weights to all positions, assigns to each position a direction from*
             * *0 to period-1, calculates the potential energies for each position and its*
             * *gradient.*


.. py:function:: build_kernel(bijection: Callable = integrators.velocity_verlet)

   Build a Periodic Orbital kernel :cite:p:`neklyudov2022orbital`.

   :param bijection: transformation used to build the orbit (given a step size).

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:class:: orbital_hmc


   Implements the (basic) user interface for the Periodic orbital MCMC kernel.

   Each iteration of the periodic orbital MCMC outputs ``period`` weighted samples from
   a single Hamiltonian orbit connecting the previous sample and momentum (latent) variable
   with precision matrix ``inverse_mass_matrix``, evaluated using the ``bijection`` as an
   integrator with discretization parameter ``step_size``.

   .. rubric:: Examples

   A new Periodic orbital MCMC kernel can be initialized and used with the following code:

   .. code::

       per_orbit = blackjax.orbital_hmc(logdensity_fn, step_size, inverse_mass_matrix, period)
       state = per_orbit.init(position)
       new_state, info = per_orbit.step(rng_key, state)

   We can JIT-compile the step function for better performance

   .. code::

       step = jax.jit(per_orbit.step)
       new_state, info = step(rng_key, state)

   :param logdensity_fn: The logarithm of the probability density function we wish to draw samples from.
   :param step_size: The value to use for the step size in for the symplectic integrator to buid the orbit.
   :param inverse_mass_matrix: The value to use for the inverse mass matrix when drawing a value for
                               the momentum and computing the kinetic energy.
   :param period: The number of steps used to build the orbit.
   :param bijection: (algorithm parameter) The symplectic integrator to use to build the orbit.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


