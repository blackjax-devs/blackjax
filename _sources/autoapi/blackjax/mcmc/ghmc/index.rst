:py:mod:`blackjax.mcmc.ghmc`
============================

.. py:module:: blackjax.mcmc.ghmc

.. autoapi-nested-parse::

   Public API for the Generalized (Non-reversible w/ persistent momentum) HMC Kernel



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.ghmc.GHMCState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.ghmc.init
   blackjax.mcmc.ghmc.kernel



.. py:class:: GHMCState



   State of the Generalized HMC algorithm.

   The Generalized HMC algorithm is persistent on its momentum, hence
   taking as input a position and momentum pair, updating and returning
   it for the next iteration. The algorithm also uses a persistent slice
   to perform a non-reversible Metropolis Hastings update, thus we also
   store the current slice variable and return its updated version after
   each iteration. To make computations more efficient, we also store
   the current logdensity as well as the current gradient of the
   logdensity.


   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: momentum
      :type: blackjax.types.PyTree

      

   .. py:attribute:: logdensity
      :type: float

      

   .. py:attribute:: logdensity_grad
      :type: blackjax.types.PyTree

      

   .. py:attribute:: slice
      :type: float

      


.. py:function:: init(rng_key: blackjax.types.PRNGKey, position: blackjax.types.PyTree, logdensity_fn: Callable)


.. py:function:: kernel(noise_fn: Callable = lambda _: 0.0, divergence_threshold: float = 1000)

   Build a Generalized HMC kernel.

   The Generalized HMC kernel performs a similar procedure to the standard HMC
   kernel with the difference of a persistent momentum variable and a non-reversible
   Metropolis-Hastings step instead of the standard Metropolis-Hastings acceptance
   step. This means that; apart from momentum and slice variables that are dependent
   on the previous momentum and slice variables, and a Metropolis-Hastings step
   performed (equivalently) as slice sampling; the standard HMC's implementation can
   be re-used to perform Generalized HMC sampling.

   :param noise_fn: A function that takes as input the slice variable and outputs a random
                    variable used as a noise correction of the persistent slice update.
                    The parameter defaults to a random variable with a single atom at 0.
   :param divergence_threshold: Value of the difference in energy above which we consider that the
                                transition is divergent.

   :returns: * *A kernel that takes a rng_key, a Pytree that contains the current state*
             * *of the chain, and free parameters of the sampling mechanism; and that*
             * *returns a new state of the chain along with information about the transition.*


