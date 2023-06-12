:py:mod:`blackjax.adaptation.pathfinder_adaptation`
===================================================

.. py:module:: blackjax.adaptation.pathfinder_adaptation

.. autoapi-nested-parse::

   Implementation of the Pathinder warmup for the HMC family of sampling algorithms.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.pathfinder_adaptation.PathfinderAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.pathfinder_adaptation.base
   blackjax.adaptation.pathfinder_adaptation.pathfinder_adaptation



.. py:class:: PathfinderAdaptationState




   .. py:attribute:: ss_state
      :type: blackjax.adaptation.step_size.DualAveragingAdaptationState

      

   .. py:attribute:: step_size
      :type: float

      

   .. py:attribute:: inverse_mass_matrix
      :type: blackjax.types.Array

      


.. py:function:: base(target_acceptance_rate: float = 0.8)

   Warmup scheme for sampling procedures based on euclidean manifold HMC.

   This adaptation runs in two steps:

   1. The Pathfinder algorithm is ran and we subsequently compute an estimate
   for the value of the inverse mass matrix, as well as a new initialization
   point for the markov chain that is supposedly closer to the typical set.
   2. We then start sampling with the MCMC algorithm and use the samples to
   adapt the value of the step size using an optimization algorithm so that
   the mcmc algorithm reaches a given target acceptance rate.

   :param target_acceptance_rate: The target acceptance rate for the step size adaptation.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.
             * *final* -- Function that returns the step size and mass matrix given a warmup state.


.. py:function:: pathfinder_adaptation(algorithm: Union[blackjax.mcmc.hmc.hmc, blackjax.mcmc.nuts.nuts], logdensity_fn: Callable, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC fmaily.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :returns: * *A function that returns the last chain state and a sampling kernel with the*
             * *tuned parameter values from an initial state.*


