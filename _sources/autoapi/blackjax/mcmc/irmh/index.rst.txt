:py:mod:`blackjax.mcmc.irmh`
============================

.. py:module:: blackjax.mcmc.irmh

.. autoapi-nested-parse::

   Public API for the Independent Rosenbluth-Metropolis-Hastings kernels.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.irmh.kernel



.. py:function:: kernel(proposal_distribution: Callable) -> Callable

   Build an Independent Random Walk Rosenbluth-Metropolis-Hastings kernel. This implies
   that the proposal distribution does not depend on the particle being mutated :cite:p:`wang2022exact`.

   :param proposal_distribution: A function that, given a PRNGKey, is able to produce a sample in the same
                                 domain of the target distribution.

   :returns: * *A kernel that takes a rng_key and a Pytree that contains the current state*
             * *of the chain and that returns a new state of the chain along with*
             * *information about the transition.*


