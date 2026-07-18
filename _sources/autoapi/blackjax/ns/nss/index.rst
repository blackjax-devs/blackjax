blackjax.ns.nss
===============

.. py:module:: blackjax.ns.nss

.. autoapi-nested-parse::

   Nested Slice Sampling (NSS) algorithm.

   An example implementation of Nested Sampling with a slice sampler as the inner
   MCMC kernel (Yallup, Kroupa & Handley, 2026, arXiv:2601.23252). The default
   :func:`build_kernel` uses hit-and-run moves shaped by the live-point covariance;
   :func:`build_swig_kernel` offers an axis-aligned slice-within-Gibbs alternative.



Functions
---------

.. autoapisummary::

   blackjax.ns.nss.covariance_proposal
   blackjax.ns.nss.coordinate_proposal
   blackjax.ns.nss.live_covariance
   blackjax.ns.nss.live_covariance_factor
   blackjax.ns.nss.live_widths
   blackjax.ns.nss.slice_constrained_step
   blackjax.ns.nss.build_kernel
   blackjax.ns.nss.coordinate_constrained_step
   blackjax.ns.nss.build_swig_kernel
   blackjax.ns.nss.as_top_level_api
   blackjax.ns.nss.swig_as_top_level_api


Module Contents
---------------

.. py:function:: covariance_proposal(init_state_fn: Callable, loglikelihood_0: blackjax.types.Array, cov: blackjax.types.Array | None = None, *, covariance_factor: blackjax.types.Array | None = None) -> Callable

   Proposal generator for nested slice sampling.

   The nested-sampling analogue of
   :func:`~blackjax.mcmc.slice.direction_proposal`: it steps along a
   covariance-shaped direction and gates the hard likelihood constraint into
   ``is_valid``. The returned ``slice_fn`` builds the candidate particle
   (recording its log-likelihood, computed once) and reports it admissible only
   when ``loglikelihood > loglikelihood_0``. Override it to write a custom
   nested stepper.

   The default NSS kernel supplies ``covariance_factor`` so the Cholesky
   factorization is shared by all inner steps. ``cov`` remains supported for
   covariance-based custom parameter callbacks and direct callers.

   :param init_state_fn: Builds a particle state from a position and birth log-likelihood.
   :param loglikelihood_0: Hard lower likelihood threshold for valid proposals.
   :param cov: Live-point covariance matrix. Used only when ``covariance_factor`` is
               not supplied.
   :param covariance_factor: Precomputed lower-triangular Cholesky factor of the live covariance.

   :rtype: A proposal generator consumed by the univariate slice kernel.


.. py:function:: coordinate_proposal(init_state_fn: Callable, loglikelihood_0: blackjax.types.Array, i: blackjax.types.Array, width: blackjax.types.Array) -> Callable

   Per-axis proposal generator for nested slice-within-Gibbs (SwiG).

   The coordinate counterpart of :func:`covariance_proposal`: it steps along
   axis ``i`` scaled by ``width`` (the direction ``width * e_i``) and gates the
   hard likelihood constraint into ``is_valid``. Like :func:`covariance_proposal`,
   the move's scale lives in the direction, so the univariate slice always runs at
   unit width. The returned ``slice_fn`` builds the candidate particle (recording
   its log-likelihood) and reports it admissible only when
   ``loglikelihood > loglikelihood_0``; it threads the full particle, so the
   recorded loglikelihood survives the sweep. Override it to write a custom
   nested coordinate stepper.


.. py:function:: live_covariance(rng_key: blackjax.types.PRNGKey, state: blackjax.ns.base.NSState, info: blackjax.ns.base.NSInfo, params: dict[str, blackjax.types.ArrayTree] | None = None) -> dict[str, blackjax.types.ArrayTree]

   Compute the live-point covariance for covariance-based custom proposals.

   :param rng_key: Unused key required by the adaptive-kernel callback protocol.
   :param state: Nested-sampling state containing the current live particles.
   :param info: Unused transition information required by the callback protocol.
   :param params: Unused previous parameters required by the callback protocol.

   :rtype: A parameter dictionary containing the live-point covariance.


.. py:function:: live_covariance_factor(rng_key: blackjax.types.PRNGKey, state: blackjax.ns.base.NSState, info: blackjax.ns.base.NSInfo, params: dict[str, blackjax.types.ArrayTree] | None = None) -> dict[str, blackjax.types.ArrayTree]

   Factor the live-point covariance once per nested-sampling step.

   :param rng_key: Unused key required by the adaptive-kernel callback protocol.
   :param state: Nested-sampling state containing the current live particles.
   :param info: Unused transition information required by the callback protocol.
   :param params: Unused previous parameters required by the callback protocol.

   :rtype: A parameter dictionary containing the lower-triangular Cholesky factor.


.. py:function:: live_widths(rng_key: blackjax.types.PRNGKey, state: blackjax.ns.base.NSState, info: blackjax.ns.base.NSInfo, params: dict[str, blackjax.types.ArrayTree] | None = None) -> dict[str, blackjax.types.ArrayTree]

   Per-axis live-point spread (std): the per-coordinate slice widths for SwiG.

   The coordinate counterpart of :func:`live_covariance_factor`: only the marginal
   per-axis spread is used, so axis correlations are deliberately ignored -- the
   defining trait of a coordinate (slice-within-Gibbs) move. Overridable via the
   ``inner_kernel_params`` seam of :func:`build_swig_kernel` and
   :func:`swig_as_top_level_api`, mirroring :func:`live_covariance_factor`.


.. py:function:: slice_constrained_step(init_state_fn: Callable, slice_kernel: Callable, proposal: Callable) -> Callable

   The slice-family constrained inner step for nested sampling.

   Runs ``slice_kernel`` with a constrained proposal generator built by
   ``proposal(init_state_fn, loglikelihood_0, **params)``; the proposal's
   ``slice_fn`` gates ``is_valid`` on the likelihood contour, so the slice
   shrinks until it lands inside it (no wasted steps). The slice counterpart to
   :func:`reject_constrained_step`, consumed by
   :func:`~blackjax.ns.from_mcmc.build_kernel`.


.. py:function:: build_kernel(init_state_fn: Callable, num_inner_steps: int, num_delete: int = 1, max_steps: int = 10, max_shrinkage: int = 100, proposal: Callable = covariance_proposal, inner_kernel_params: Callable | None = None) -> Callable

   Build the Nested Slice Sampling kernel.

   :param init_state_fn: Builds a particle state from a position and birth log-likelihood.
   :param num_inner_steps: Number of slice steps per new particle. Prefer
                           ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing (bare ``dim`` is
                           the minimum; see :func:`as_top_level_api`).
   :param num_delete: Number of particles deleted and replaced per step (default 1).
   :param max_steps: Cap on stepping-out expansions per slice (default 10).
   :param max_shrinkage: Cap on shrinkage evaluations per slice (default 100).
   :param proposal: Proposal factory ``(init_state_fn, loglikelihood_0, **params) ->
                    proposal_generator`` (:func:`covariance_proposal` by default). The
                    default proposal consumes a precomputed ``covariance_factor``. Override
                    to write a custom nested stepper.
   :param inner_kernel_params: Computes the inner-kernel parameters from the live points each step,
                               ``(rng_key, state, info, params) -> params``. When ``None``, uses
                               :func:`live_covariance_factor` with the default proposal and
                               :func:`live_covariance` with a custom proposal, preserving the existing
                               covariance-based extension seam.

   :rtype: A kernel ``kernel(rng_key, state)`` that returns ``(new_state, info)``.


.. py:function:: coordinate_constrained_step(init_state_fn: Callable, slice_kernel: Callable, proposal: Callable = coordinate_proposal, coordinate_order: Callable = random_order) -> Callable

   The coordinate-sweep constrained inner step for nested sampling (SwiG).

   The slice-within-Gibbs counterpart of :func:`slice_constrained_step`: one call
   sweeps every axis once -- in the order set by ``coordinate_order`` -- updating
   each by a univariate slice from the per-axis proposal generator
   ``proposal(init_state_fn, loglikelihood_0, i, width)`` (:func:`coordinate_proposal`
   by default, the axis analogue of :func:`covariance_proposal` passed to
   :func:`slice_constrained_step`), which steps along ``width * e_i`` and gates the
   likelihood contour into ``is_valid``. As with the hit-and-run path the scale
   lives in the direction, so the univariate slice runs at unit width. Consumed by
   :func:`~blackjax.ns.from_mcmc.build_kernel` exactly like the hit-and-run step.


.. py:function:: build_swig_kernel(init_state_fn: Callable, num_inner_steps: int, num_delete: int = 1, max_steps: int = 10, max_shrinkage: int = 100, proposal: Callable = coordinate_proposal, coordinate_order: Callable = random_order, inner_kernel_params: Callable = live_widths) -> Callable

   Build the Nested Slice-within-Gibbs (SwiG) kernel.

   The coordinate counterpart of :func:`build_kernel`: each inner step is a full
   coordinate *sweep* rather than a hit-and-run move. Axes are visited in the
   order set by ``coordinate_order`` (:func:`~blackjax.mcmc.slice.random_order`
   by default, or :func:`~blackjax.mcmc.slice.fixed_order`), and each is updated
   by a univariate slice gated on the likelihood contour and scaled by that
   axis's live width (the per-axis spread of the live points; correlations are
   ignored). Prefer this when the target is close to axis-aligned, or when its
   correlations are unreliable to estimate. Pair with
   :func:`swig_as_top_level_api` for the bundled (init, step) algorithm.

   :param init_state_fn: Builds a particle state from a position and birth log-likelihood.
   :param num_inner_steps: Number of coordinate sweeps per new particle. Prefer
                           ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing (bare ``dim`` is
                           the minimum; see :func:`swig_as_top_level_api`).
   :param num_delete: Number of particles deleted and replaced per step (default 1).
   :param max_steps: Cap on stepping-out expansions per univariate slice (default 10).
   :param max_shrinkage: Cap on shrinkage evaluations per univariate slice (default 100).
   :param proposal: Per-axis proposal factory ``(init_state_fn, loglikelihood_0, i, width)
                    -> proposal_generator`` (:func:`coordinate_proposal` by default). The
                    coordinate analogue of the ``proposal`` seam on :func:`build_kernel`.
   :param coordinate_order: Sweep-order primitive ``(rng_key, d) -> indices``
                            (:func:`~blackjax.mcmc.slice.random_order` by default).
   :param inner_kernel_params: Computes the inner-kernel parameters from the live points each step,
                               ``(rng_key, state, info, params) -> params`` (:func:`live_widths` by
                               default, the per-axis live-point spread).

   :rtype: A kernel ``kernel(rng_key, state)`` that returns ``(new_state, info)``.


.. py:function:: as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, num_inner_steps: int, num_delete: int = 1, max_steps: int = 10, max_shrinkage: int = 100, proposal: Callable = covariance_proposal, inner_kernel_params: Callable | None = None) -> blackjax.SamplingAlgorithm

   Creates a Nested Slice Sampling (NSS) algorithm, ``blackjax.nss``.

   Nested Sampling with a hit-and-run slice inner kernel: each particle
   replacement runs ``num_inner_steps`` constrained slice moves along
   directions shaped by the live-point covariance.

   :param logprior_fn: Log-prior of a single particle.
   :param loglikelihood_fn: Log-likelihood of a single particle.
   :param num_inner_steps: Number of slice steps per new particle. Use
                           ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing within the
                           likelihood contour; bare ``dim`` is the minimum and can bias the evidence
                           *upward* for ``dim > 10`` (the inner chain must decorrelate the new particle
                           from the deleted one, not merely satisfy the constraint).
   :param num_delete: Number of particles deleted and replaced per step (default 1).
   :param max_steps: Cap on stepping-out expansions per slice (default 10).
   :param max_shrinkage: Cap on shrinkage evaluations per slice (default 100).
   :param proposal: Proposal factory ``(init_state_fn, loglikelihood_0, **params) ->
                    proposal_generator`` (:func:`covariance_proposal` by default). The
                    default proposal consumes a precomputed ``covariance_factor``. Override
                    to write a custom nested stepper.
   :param inner_kernel_params: Computes the inner-kernel parameters from the live points,
                               ``(rng_key, state, info, params) -> params``. When ``None``, uses
                               :func:`live_covariance_factor` with the default proposal and
                               :func:`live_covariance` with a custom proposal. Used both to seed
                               ``init`` and to update each step.

   :returns: * A ``SamplingAlgorithm`` whose ``step(rng_key, state)`` returns
             * ``(new_state, info)``.

   .. rubric:: Notes

   The live particles in the run state (``state.particles``) are **not** posterior
   samples: they are the current likelihood shell and at termination collapse to
   the highest-likelihood mode. For correctly-weighted posterior draws, pass the
   dead points through :func:`~blackjax.ns.utils.finalise` and resample with
   :func:`~blackjax.ns.utils.sample`.

   The covariance-shaped proposal bridges between modes only up to moderate
   separation. For strongly multimodal targets, ensure the initial live points
   span every mode (and consider a clustering inner kernel); minor modes are still
   weighted correctly in the resampled posterior, but may be absent from the final
   live set.


.. py:function:: swig_as_top_level_api(logprior_fn: Callable, loglikelihood_fn: Callable, num_inner_steps: int, num_delete: int = 1, max_steps: int = 10, max_shrinkage: int = 100, proposal: Callable = coordinate_proposal, coordinate_order: Callable = random_order, inner_kernel_params: Callable = live_widths) -> blackjax.SamplingAlgorithm

   Creates a Nested Slice-within-Gibbs (SwiG) sampling algorithm, ``blackjax.nsswig``.

   Nested Sampling with an axis-aligned slice-within-Gibbs inner kernel: each
   particle replacement runs ``num_inner_steps`` constrained coordinate sweeps,
   each axis scaled by the live-point spread (correlations are ignored). The
   coordinate counterpart of :func:`as_top_level_api`; prefer it when the target
   is close to axis-aligned or its correlations are unreliable to estimate.

   :param logprior_fn: Log-prior of a single particle.
   :param loglikelihood_fn: Log-likelihood of a single particle.
   :param num_inner_steps: Number of coordinate sweeps per new particle. Use
                           ``num_inner_steps >= max(5, 2 * dim)`` for reliable mixing within the
                           likelihood contour; bare ``dim`` is the minimum and can bias the evidence
                           *upward* for ``dim > 10`` (the inner chain must decorrelate the new particle
                           from the deleted one, not merely satisfy the constraint).
   :param num_delete: Number of particles deleted and replaced per step (default 1).
   :param max_steps: Cap on stepping-out expansions per univariate slice (default 10).
   :param max_shrinkage: Cap on shrinkage evaluations per univariate slice (default 100).
   :param proposal: Per-axis proposal factory ``(init_state_fn, loglikelihood_0, i, width)
                    -> proposal_generator`` (:func:`coordinate_proposal` by default), the
                    coordinate analogue of the ``proposal`` seam on :func:`as_top_level_api`.
   :param coordinate_order: Sweep-order primitive ``(rng_key, d) -> indices``
                            (:func:`~blackjax.mcmc.slice.random_order` by default).
   :param inner_kernel_params: Computes the inner-kernel parameters from the live points,
                               ``(rng_key, state, info, params) -> params`` (:func:`live_widths` by
                               default). Used both to seed ``init`` and to update each step.

   :returns: * A ``SamplingAlgorithm`` whose ``step(rng_key, state)`` returns
             * ``(new_state, info)``.

   .. rubric:: Notes

   The live particles in the run state (``state.particles``) are **not** posterior
   samples: they are the current likelihood shell and at termination collapse to
   the highest-likelihood mode. For correctly-weighted posterior draws, pass the
   dead points through :func:`~blackjax.ns.utils.finalise` and resample with
   :func:`~blackjax.ns.utils.sample`.

   For strongly multimodal targets, ensure the initial live points span every mode
   (the axis-aligned per-particle proposal does not bridge well-separated modes);
   minor modes are still weighted correctly in the resampled posterior, but may be
   absent from the final live set.


