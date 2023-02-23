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
   blackjax.mcmc.periodic_orbital.PeriodicOrbitalInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.periodic_orbital.init
   blackjax.mcmc.periodic_orbital.kernel
   blackjax.mcmc.periodic_orbital.periodic_orbital_proposal



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
      :type: blackjax.types.PyTree

      

   .. py:attribute:: weights
      :type: blackjax.types.Array

      

   .. py:attribute:: directions
      :type: blackjax.types.Array

      

   .. py:attribute:: logdensities
      :type: blackjax.types.Array

      

   .. py:attribute:: logdensities_grad
      :type: blackjax.types.PyTree

      


.. py:class:: PeriodicOrbitalInfo



   Additional information on the states in the orbit.

   This additional information can be used for debugging or computing
   diagnostics.

   momentum
       the momentum that was sampled and used to integrate the trajectory.
   weights_mean
       mean of the the unnormalized weights of the orbit, ideally close
       to the (unknown) constant of proportionally missing from the target.
   weights_variance
       variance of the unnormalized weights of the orbit, ideally close to 0.

   .. py:attribute:: momentums
      :type: blackjax.types.PyTree

      

   .. py:attribute:: weights_mean
      :type: float

      

   .. py:attribute:: weights_variance
      :type: float

      


.. py:function:: init(position: blackjax.types.PyTree, logdensity_fn: Callable, period: int) -> PeriodicOrbitalState

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


.. py:function:: kernel(bijection: Callable = integrators.velocity_verlet)

   Build a Periodic Orbital kernel :cite:p:`neklyudov2022orbital`.

   :param bijection: transformation used to build the orbit (given a step size).

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


.. py:function:: periodic_orbital_proposal(bijection: Callable, kinetic_energy_fn: Callable, period: int, step_size: float) -> Callable

   Periodic Orbital algorithm.

   The algorithm builds and orbit and computes the weights for each of its steps
   by applying a bijection `period` times, both forwards and backwards depending
   on the direction of the initial state.

   :param bijection: continuous, differentialble and bijective transformation used to build
                     the orbit step by step.
   :param kinetic_energy_fn: function that computes the kinetic energy.
   :param period: total steps used to build the orbit.
   :param step_size: size between each step of the orbit.

   :returns: * *A kernel that generates a new periodic orbital state and information*
             * *about the transition.*


