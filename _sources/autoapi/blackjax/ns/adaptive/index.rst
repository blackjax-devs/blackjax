blackjax.ns.adaptive
====================

.. py:module:: blackjax.ns.adaptive

.. autoapi-nested-parse::

   Adaptive Nested Sampling for BlackJAX.

   It combines the NS analogue of SMC's adaptive (tempering) schedule -- evidence
   integration via ``NSIntegrator`` -- with inner-kernel tuning, wrapping the live
   particles, the integrator, and the inner-kernel parameters into ``AdaptiveNSState``.



Functions
---------

.. autoapisummary::

   blackjax.ns.adaptive.init
   blackjax.ns.adaptive.build_kernel


Module Contents
---------------

.. py:function:: init(positions: blackjax.types.ArrayLikeTree, init_state_fn: Callable, loglikelihood_birth: float = jnp.nan, update_inner_kernel_params_fn: Callable | None = None, rng_key: blackjax.types.PRNGKey | None = None) -> AdaptiveNSState

   Initialize the adaptive Nested Sampling state from live positions.

   :param positions: Initial positions of the live particles.
   :param init_state_fn: Maps positions to a ``StateWithLogLikelihood`` (typically vmapped).
   :param loglikelihood_birth: Birth log-likelihood assigned to the initial particles.
   :param update_inner_kernel_params_fn: Optional ``(rng_key, state, info, params) -> params`` used to seed the
                                         inner kernel parameters; if ``None`` the parameters start empty.
   :param rng_key: PRNG key passed to ``update_inner_kernel_params_fn``.

   :rtype: The initial ``AdaptiveNSState``.


.. py:function:: build_kernel(delete_fn: Callable, inner_kernel: Callable, update_inner_kernel_params_fn: Callable[[blackjax.types.PRNGKey, blackjax.ns.base.NSState, blackjax.ns.base.NSInfo, dict[str, blackjax.types.ArrayTree]], dict[str, blackjax.types.ArrayTree]]) -> Callable

   Build an adaptive Nested Sampling kernel.

   Each step runs the inner kernel with the parameters carried in the incoming
   state (computed by the previous step's update), then recomputes those
   parameters from the resulting state and this step's update ``info`` for use
   on the next step, and advances the evidence integrator.

   :param delete_fn: Selects which live particles to delete and replace each step.
   :param inner_kernel: Inner MCMC kernel used to generate replacement particles, called with
                        the current ``inner_kernel_params``.
   :param update_inner_kernel_params_fn: ``(rng_key, state, info, params) -> params`` recomputing the inner
                                         kernel parameters after each step.

   :rtype: A kernel ``(rng_key, AdaptiveNSState) -> (AdaptiveNSState, NSInfo)``.


