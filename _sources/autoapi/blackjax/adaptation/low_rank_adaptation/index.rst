blackjax.adaptation.low_rank_adaptation
=======================================

.. py:module:: blackjax.adaptation.low_rank_adaptation

.. autoapi-nested-parse::

   Adaptation of the low-rank-modified mass matrix for HMC-family samplers.

   Implements Algorithm 1 of :cite:p:`sountsov2025preconditioning`, following the
   nutpie reference implementation.  The mass matrix has the form

   .. math::

       M^{-1} = \operatorname{diag}(\sigma)
                \bigl(I + U(\Lambda - I)U^\top\bigr)
                \operatorname{diag}(\sigma)

   and is adapted by minimising the sample Fisher divergence.  All HMC operations
   cost :math:`O(dk)` where :math:`k` is the low rank.

   Key algorithmic choices that match nutpie:

   * **Population variance** (divide by *n*, not *n-1*) for diagonal scaling.
   * **σ clipping** to ``[1e-20, 1e20]`` to avoid premature saturation.
   * **Optimal translation** μ* = x̄ + σ²⊙ᾱ is computed and returned.
   * **Regularisation**: projected covariance is ``P P^T / (n·γ) + I``
     (nutpie's convention; default γ=1 gives ``P P^T / n + I``).
   * **SPD mean** via eigendecomposition of the gradient covariance (not
     Cholesky of the draw covariance).
   * **Eigenvalue masking**: components with λ ∈ [1/cutoff, cutoff] are set
     to λ=1 rather than clipped (default cutoff=2, matching nutpie's ``c=2``).

   The warmup schedule mirrors Stan's window adaptation: an initial fast phase,
   a series of doubling slow windows (metric + step-size), and a final fast
   phase.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.low_rank_adaptation.LowRankAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.low_rank_adaptation.base
   blackjax.adaptation.low_rank_adaptation.low_rank_window_adaptation


Module Contents
---------------

.. py:class:: LowRankAdaptationState



   State for the low-rank mass matrix window adaptation.

   ss_state
       Internal state of the dual-averaging step-size adapter.
   sigma
       Current diagonal scaling, shape ``(d,)``.
   mu_star
       Current optimal translation ``x̄ + σ² ⊙ ᾱ``, shape ``(d,)``.
   U
       Current low-rank eigenvectors, shape ``(d, max_rank)``.
   lam
       Current eigenvalues, shape ``(max_rank,)``.
   step_size
       Current step size (updated every iteration).
   draws_buffer
       Circular buffer storing the last ``buffer_size`` chain positions,
       shape ``(buffer_size, d)``.
   grads_buffer
       Circular buffer storing the corresponding log-density gradients,
       shape ``(buffer_size, d)``.
   buffer_idx
       Number of samples written to the current buffer (resets at each slow
       window boundary).


   .. py:attribute:: ss_state
      :type:  blackjax.adaptation.step_size.DualAveragingAdaptationState


   .. py:attribute:: sigma
      :type:  blackjax.types.Array


   .. py:attribute:: mu_star
      :type:  blackjax.types.Array


   .. py:attribute:: U
      :type:  blackjax.types.Array


   .. py:attribute:: lam
      :type:  blackjax.types.Array


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: draws_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: grads_buffer
      :type:  blackjax.types.Array


   .. py:attribute:: buffer_idx
      :type:  int


.. py:function:: base(max_rank: int = 10, target_acceptance_rate: float = 0.8, gamma: float = 1.0, cutoff: float = 2.0) -> tuple[Callable, Callable, Callable]

   Warmup scheme using the low-rank mass matrix adaptation.

   Mirrors Stan's three-phase schedule but replaces Welford covariance
   estimation with the Fisher-divergence-minimising low-rank metric of
   :cite:p:`sountsov2025preconditioning`, following nutpie's implementation.

   :param max_rank: Maximum number of eigenvectors retained in the low-rank correction.
   :param target_acceptance_rate: Target acceptance rate for dual-averaging step-size adaptation.
   :param gamma: Regularisation scale.  The projected covariance is divided by
                 ``n * gamma`` before adding identity (nutpie convention).  Default
                 ``1.0`` gives ``C = P P^T / n + I``.
   :param cutoff: Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked
                  (eigenvalue set to 1).  Default ``2.0`` matches nutpie's ``c=2``.

   :returns: The three adaptation primitives expected by the window-adaptation loop.
   :rtype: init, update, final


.. py:function:: low_rank_window_adaptation(algorithm, logdensity_fn: Callable, max_rank: int = 10, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, gamma: float = 1.0, cutoff: float = 2.0, progress_bar: bool = False, adaptation_info_fn: Callable = return_all_adapt_info, integrator=mcmc.integrators.velocity_verlet, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt step size and a low-rank mass matrix for HMC-family samplers.

   Uses the three-phase Stan-style warmup schedule while replacing Welford
   covariance estimation with the Fisher-divergence-minimising low-rank
   metric of :cite:p:`sountsov2025preconditioning`.

   The returned ``AdaptationAlgorithm`` has a single ``run`` method::

       (state, params), info = warmup.run(rng_key, position, num_steps=1000)
       nuts = blackjax.nuts(logdensity_fn, **params)

   :param algorithm: An HMC-family algorithm object (e.g. ``blackjax.nuts``).
   :param logdensity_fn: Log-density of the target distribution.
   :param max_rank: Maximum number of eigenvectors in the low-rank correction.
   :param initial_step_size: Starting step size (adapted automatically).
   :param target_acceptance_rate: Target acceptance rate for dual averaging.
   :param gamma: Regularisation scale; projected covariance is divided by ``n * gamma``
                 before adding identity (nutpie convention).
   :param cutoff: Eigenvectors with eigenvalue in ``[1/cutoff, cutoff]`` are masked.
                  Default ``2.0`` matches nutpie's ``c=2``.
   :param progress_bar: Show a progress bar during warmup.
   :param adaptation_info_fn: Controls what adaptation info is retained; see
                              ``blackjax.adaptation.base``.
   :param integrator: Integrator to pass to ``algorithm.build_kernel``.
   :param \*\*extra_parameters: Additional keyword arguments forwarded to the kernel at every step
                                (e.g. ``num_integration_steps`` for HMC).

   :returns: * An ``AdaptationAlgorithm`` whose ``run`` method returns
             * ``(AdaptationResults, info)``.  ``AdaptationResults.parameters`` contains
             * ``step_size``, ``inverse_mass_matrix`` (a :func:`gaussian_euclidean_low_rank`
             * ``Metric`` object), and any ``extra_parameters``.
             * ``AdaptationResults.state`` is re-initialised at the optimal translation
             * *μ* = x̄ + σ²⊙ᾱ, so it can be passed directly as the starting state for*
             * *production sampling.  The last chain state from warmup is available as*
             * ``warmup_info[-1].state``, and μ* as
             * ``warmup_info[-1].adaptation_state.mu_star``.


