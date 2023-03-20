:py:mod:`blackjax.mcmc.hmc`
===========================

.. py:module:: blackjax.mcmc.hmc

.. autoapi-nested-parse::

   Public API for the HMC Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.hmc.HMCState
   blackjax.mcmc.hmc.HMCInfo



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.hmc.init
   blackjax.mcmc.hmc.kernel



.. py:class:: HMCState



   State of the HMC algorithm.

   The HMC algorithm takes one position of the chain and returns another
   position. In order to make computations more efficient, we also store
   the current logdensity as well as the current gradient of the logdensity.


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.PyTree

      


.. py:class:: HMCInfo



   Additional information on the HMC transition.

   This additional information can be used for debugging or computing
   diagnostics.

   momentum:
       The momentum that was sampled and used to integrate the trajectory.
   acceptance_rate
       The acceptance probability of the transition, linked to the energy
       difference between the original and the proposed states.
   is_accepted
       Whether the proposed position was accepted or the original position
       was returned.
   is_divergent
       Whether the difference in energy between the original and the new state
       exceeded the divergence threshold.
   energy:
       Total energy of the transition.
   proposal
       The state proposed by the proposal. Typically includes the position and
       momentum.
   step_size
       Size of the integration step.
   num_integration_steps
       Number of times we run the symplectic integrator to build the trajectory


   .. py:attribute:: momentum
      :type: blackjax.types.PyTree

      

   .. py:attribute:: acceptance_rate
      :type: float

      

   .. py:attribute:: is_accepted
      :type: bool

      

   .. py:attribute:: is_divergent
      :type: bool

      

   .. py:attribute:: energy
      :type: float

      

   .. py:attribute:: proposal
      :type: blackjax.mcmc.integrators.IntegratorState

      

   .. py:attribute:: num_integration_steps
      :type: int

      


.. py:function:: init(position: blackjax.types.PyTree, logdensity_fn: Callable)


.. py:function:: kernel(integrator: Callable = integrators.velocity_verlet, divergence_threshold: float = 1000)

   Build a HMC kernel.

   :param integrator: The symplectic integrator to use to integrate the Hamiltonian dynamics.
   :param divergence_threshold: Value of the difference in energy above which we consider that the transition is divergent.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


