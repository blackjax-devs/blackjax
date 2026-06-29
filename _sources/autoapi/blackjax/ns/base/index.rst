blackjax.ns.base
================

.. py:module:: blackjax.ns.base

.. autoapi-nested-parse::

   Base components for Nested Sampling.

   Defines the particle state carrying loglikelihood information, a generic kernel
   builder that deletes the lowest-likelihood particles and replaces them with an
   inner kernel, and the default deletion strategy selecting the ``num_delete``
   particles with the lowest loglikelihoods.

   .. rubric:: References

   .. [1] Skilling, J. (2006). "Nested sampling for general Bayesian computation."
          Bayesian Analysis, 1(4), 833-859. https://doi.org/10.1214/06-BA127



Classes
-------

.. autoapisummary::

   blackjax.ns.base.NSState
   blackjax.ns.base.NSInfo


Functions
---------

.. autoapisummary::

   blackjax.ns.base.init
   blackjax.ns.base.build_kernel
   blackjax.ns.base.delete_fn


Module Contents
---------------

.. py:class:: NSState



   State of the Nested Sampler.

   At the most basic level, this is just a wrapper around a ``StateWithLogLikelihood``;
   richer NS implementations (e.g. ``AdaptiveNSState``) carry extra fields.

   .. attribute:: particles

      The ``StateWithLogLikelihood`` of the current live particles.


   .. py:attribute:: particles
      :type:  StateWithLogLikelihood


.. py:class:: NSInfo



   Additional information returned at each step of the Nested Sampling algorithm.

   .. attribute:: particles

      The StateWithLogLikelihood of particles that were marked as "dead" (replaced).

   .. attribute:: update_info

      A NamedTuple (or any PyTree) containing information from the update step
      (inner kernel) used to generate new live particles.


   .. py:attribute:: particles
      :type:  StateWithLogLikelihood


   .. py:attribute:: update_info
      :type:  NamedTuple


.. py:function:: init(positions: blackjax.types.ArrayLikeTree, init_state_fn: Callable, loglikelihood_birth: float = jnp.nan) -> NSState

   Initializes the Nested Sampler state.

   :param positions: An initial set of positions (PyTree of arrays) drawn from the prior
                     distribution. The leading dimension of each leaf array must be equal to
                     the number of positions.
   :param init_state_fn: A function that builds the particle state (``StateWithLogLikelihood``)
                         from positions; ``init`` wraps the result in an ``NSState``. Typically
                         vmapped over the live set.
   :param loglikelihood_birth: The initial log-likelihood birth threshold. Defaults to NaN, which
                               implies no initial likelihood constraint beyond the prior.

   :returns: The initial state of the Nested Sampler.
   :rtype: NSState


.. py:function:: build_kernel(delete_fn: Callable, inner_kernel: Callable) -> Callable

   Build a generic Nested Sampling kernel.

   This function creates a kernel for the Nested Sampling algorithm by combining
   a particle deletion function and an inner kernel for generating new particles.

   :param delete_fn: A deletion function, typically partially applied with ``num_delete``,
                     with effective signature ``(state) -> (dead_idx, target_update_idx)``.
                     Receives the full NS state (duck-typed) and identifies particles
                     to be deleted and the indices to update.
   :param inner_kernel: A kernel function with the signature
                        ``(rng_key, state, loglikelihood_0) -> (new_particles, info)``
                        that generates replacement particles. Receives the full NS state
                        (duck-typed) and a single PRNG key; returns a
                        ``StateWithLogLikelihood`` with leading dimension ``num_delete``.
                        The number of particles to produce is known at construction time.

   :returns: A kernel function for Nested Sampling:
             ``(rng_key, state) -> (new_state, ns_info)``.
   :rtype: Callable


.. py:function:: delete_fn(state: NSState, num_delete: int) -> tuple[blackjax.types.Array, blackjax.types.Array]

   Identifies particles to be deleted.

   Selects the ``num_delete`` particles with the lowest log-likelihoods
   and marks them as "dead".

   :param state: The current NS state (duck-typed; must have ``.particles.loglikelihood``).
   :param num_delete: The number of particles to delete and subsequently replace.

   :returns: * A tuple ``(dead_idx, target_update_idx)`` of indices marked for deletion
             * *and of slots to overwrite (identical here).*


