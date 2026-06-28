blackjax.mcmc.slice
===================

.. py:module:: blackjax.mcmc.slice

.. autoapi-nested-parse::

   Public API for the Slice sampling family.

   Every slice update is univariate: a one-dimensional slice through the current
   point. Multivariate behaviour is determined entirely by the proposal generator
   that produces the line, ``proposal_generator(rng_key, position, logdensity_fn)
   -> slice_fn`` with ``slice_fn(t) -> (state, is_valid)``. The candidate state is
   threaded straight through the kernel, so a proposal can record extra quantities
   on it.

   Two samplers are built on this spine:

   1. Multivariate slice: one univariate slice along a random direction. This is
      the top-level ``blackjax.slice_sampling`` (:func:`as_top_level_api`), with
      the direction drawn by :func:`direction_proposal` (a ``scale``-shaped random
      direction, unit by default). Chaining such random-direction moves is the
      hit-and-run strategy.

   2. Coordinate-wise (slice-within-Gibbs, :func:`coordinate_slice`): sweep the
      coordinate axes in turn, updating each full conditional with a univariate
      slice.

   The one-dimensional interval is built by the stepping-out or doubling procedure
   of Neal (2003), passed as a callable (``interval=stepping_out`` or
   ``interval=doubling``), then narrowed by shrinkage to draw the new point.
   Doubling additionally applies the Fig. 6 acceptance test. Additional
   constraints are not built in but added downstream by overriding the proposal,
   which gates on ``is_valid`` and may record extra quantities on the state.

   .. rubric:: References

   .. [1] Radford M. Neal, "Slice sampling", The Annals of Statistics,
      Ann. Statist. 31(3), 705-767, (June 2003).



Classes
-------

.. autoapisummary::

   blackjax.mcmc.slice.SliceState
   blackjax.mcmc.slice.SliceInfo


Functions
---------

.. autoapisummary::

   blackjax.mcmc.slice.init
   blackjax.mcmc.slice.stepping_out
   blackjax.mcmc.slice.doubling
   blackjax.mcmc.slice.build_kernel
   blackjax.mcmc.slice.random_order
   blackjax.mcmc.slice.fixed_order
   blackjax.mcmc.slice.coordinate_proposal
   blackjax.mcmc.slice.build_coordinate_kernel
   blackjax.mcmc.slice.sample_direction
   blackjax.mcmc.slice.direction_proposal
   blackjax.mcmc.slice.as_top_level_api
   blackjax.mcmc.slice.coordinate_slice


Module Contents
---------------

.. py:class:: SliceState



   State of the Slice sampling chain.

   position
       Current position of the chain.
   logdensity
       Log-density of the target at ``position``.



   .. py:attribute:: position
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: logdensity
      :type:  float


.. py:class:: SliceInfo



   Additional information on a Slice sampling transition.

   is_accepted
       Whether shrinkage found a valid point within ``max_shrinkage`` steps.
       Always ``True`` for an unconstrained target (the slice always contains
       the current point); can be ``False`` when the proposal gates a
       constraint into ``is_valid`` and the budget is exhausted, leaving the
       chain in place.
       For the coordinate sweep it is ``True`` only if every coordinate
       succeeded.
   num_expansions
       Number of interval expansions (stepping-out steps or doublings).
       Summed over coordinates for the sweep.
   num_shrink
       Number of shrinkage evaluations taken to find the new point.
       Summed over coordinates for the sweep.
   bracket_left, bracket_right
       The realized slice bracket in the 1-D slice coordinate ``t``, where the
       current point sits at ``t = 0`` (so typically
       ``bracket_left <= 0 <= bracket_right``). For the multivariate slice these
       are scalars; for the coordinate sweep they are per-axis ``t`` values, a
       PyTree aligned with ``position``. The bracket width is
       ``bracket_right - bracket_left``.



   .. py:attribute:: is_accepted
      :type:  blackjax.types.Array


   .. py:attribute:: num_expansions
      :type:  blackjax.types.Array


   .. py:attribute:: num_shrink
      :type:  blackjax.types.Array


   .. py:attribute:: bracket_left
      :type:  blackjax.types.ArrayTree


   .. py:attribute:: bracket_right
      :type:  blackjax.types.ArrayTree


.. py:function:: init(position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable) -> SliceState

   Create an initial state from a position and log-density function.


.. py:function:: stepping_out(rng_key: blackjax.types.PRNGKey, in_slice: Callable, width: float, max_expansions: int) -> tuple[blackjax.types.Array, blackjax.types.Array, blackjax.types.Array, AcceptFn]

   Neal (2003) Fig. 3 stepping-out interval, in t-space (x0 at t=0).

   An interval procedure is a pluggable callable (pass it as
   ``interval=stepping_out``). It returns its own acceptance test so the kernel
   never branches on a name; stepping-out needs none, so ``accept_fn`` always
   returns ``True``.

   :returns: * **The tuple ``(left, right, num_expansions, accept_fn)``** (*the bracket*)
             * *endpoints, the number of expansions, and the acceptance test.*


.. py:function:: doubling(rng_key: blackjax.types.PRNGKey, in_slice: Callable, width: float, max_expansions: int) -> tuple[blackjax.types.Array, blackjax.types.Array, blackjax.types.Array, AcceptFn]

   Neal (2003) Fig. 4 doubling interval, vectorized, in t-space.

   Expands one (randomly chosen) side at a time, doubling the bracket each
   step, until both ends are outside the slice or ``max_expansions`` is hit. A
   pluggable interval callable, like :func:`stepping_out`. Its ``accept_fn`` is
   Neal's Fig. 6 acceptance test bound to this (original) bracket, which is
   required for doubling's reversibility.

   :returns: * **The tuple ``(left, right, num_expansions, accept_fn)``** (*the bracket*)
             * *endpoints, the number of expansions, and the acceptance test.*


.. py:function:: build_kernel(interval: Callable = doubling, max_expansions: int = 10, max_shrinkage: int = 100) -> Callable

   Build a slice kernel driven by a proposal generator.

   The kernel performs one univariate slice using ``proposal_generator``, a
   callable ``(rng_key, position, logdensity_fn) -> slice_fn`` where
   ``slice_fn(t) -> (state, is_valid)`` builds the candidate state at coordinate
   ``t`` and reports whether it is admissible. Because the candidate state is
   threaded straight out, the proposal can record extra quantities on it and
   consume a constraint through ``is_valid``. To sample under a constraint,
   override the proposal generator rather than the kernel.

   :param interval: Interval-finding procedure, passed directly as a callable. Use
                    :func:`doubling` (the default, Neal Fig. 4 with the Fig. 6 acceptance
                    test) or :func:`stepping_out` (Neal Fig. 3).
   :param max_expansions: Cap on interval expansions (doublings or stepping-out steps).
   :param max_shrinkage: Cap on shrinkage evaluations. Bounds the loop; on exhaustion the chain
                         stays put.

   :returns: * *A kernel that takes a rng_key, the current state, a log-density function, a*
             * *proposal generator and a bracket width, and returns a new state along with*
             * *information about the transition.*


.. py:function:: random_order(rng_key: blackjax.types.PRNGKey, d: int) -> blackjax.types.Array

   A fresh random permutation of the ``d`` coordinate indices (the default).


.. py:function:: fixed_order(rng_key: blackjax.types.PRNGKey, d: int) -> blackjax.types.Array

   Sweep the coordinates in fixed natural order ``0, 1, ..., d - 1``.


.. py:function:: coordinate_proposal(rng_key: blackjax.types.PRNGKey, position: blackjax.types.ArrayLikeTree, logdensity_fn: Callable, i: int) -> Callable

   Default per-axis proposal for the coordinate sweep.

   The coordinate analogue of :func:`direction_proposal`: a unit step along
   flattened axis ``i`` (the one-hot direction ``e_i``), so ``x(t)`` is
   ``position`` with ``flat[i] += t`` and the current point sits at ``t = 0``.
   Shares the ``slice_fn(t) -> (state, is_valid)`` contract of the multivariate
   proposals.

   A constraint is added the same way as on the multivariate path -- by
   overriding the proposal (``axis_proposal``) to gate ``is_valid``; there is no
   built-in constraint argument.


.. py:function:: build_coordinate_kernel(interval: Callable = doubling, axis_proposal: Callable = coordinate_proposal, coordinate_order: Callable = random_order, initial_widths: float | blackjax.types.Array = 1.0, max_expansions: int = 10, max_shrinkage: int = 100) -> Callable

   Build a coordinate-wise (slice-within-Gibbs) kernel.

   One step updates each scalar coordinate's full conditional with a univariate
   slice, in the order given by ``coordinate_order``, the choice function
   ``(rng_key, d) -> indices`` (:func:`random_order`, the default, or
   :func:`fixed_order`). Each coordinate move is drawn by ``axis_proposal``, the
   per-axis analogue of the multivariate ``proposal_generator``
   (:func:`coordinate_proposal` by default); override it to gate a constraint
   into ``is_valid``. ``initial_widths`` is a scalar (applied to every
   coordinate) or a length-``D`` array of per-coordinate bracket widths.

   :returns: * *A kernel that takes a rng_key, the current state and a log-density function,*
             * *and returns a new state along with information about the transition.*


.. py:function:: sample_direction(rng_key: blackjax.types.PRNGKey, position: blackjax.types.ArrayLikeTree, scale: float | blackjax.types.Array = 1.0) -> blackjax.types.ArrayTree

   A random slice direction shaped by ``scale`` and normalized to unit length.

   ``scale`` is a scalar (isotropic), a vector (per-coordinate / diagonal) or a
   dense matrix (a full preconditioner, applied as a linear map to
   standard-normal noise, so its covariance is ``scale @ scale.T``). Defaults
   to ``1.0`` (uniformly random unit directions).


.. py:function:: direction_proposal(scale: float | blackjax.types.Array = 1.0) -> Callable

   Proposal-generator factory: slice along a random ``scale``-shaped direction.

   See :func:`sample_direction` for ``scale`` (scalar / vector / dense, unit by
   default). Pass as
   ``slice_sampling(logp, proposal_generator=direction_proposal(scale))``.


.. py:function:: as_top_level_api(logdensity_fn: Callable, *, proposal_generator: Callable = direction_proposal(), width: float = 1.0, interval: Callable = doubling, max_expansions: int = 10, max_shrinkage: int = 100) -> blackjax.base.SamplingAlgorithm

   Multivariate slice sampler, ``blackjax.slice_sampling``.

   Each step takes one univariate slice along a random direction (chaining such
   moves is the hit-and-run strategy) drawn by ``proposal_generator``. The
   default :func:`direction_proposal` draws a uniformly random direction; pass
   ``direction_proposal(scale)`` to precondition, or override with your own
   proposal to gate a constraint or record extra quantities on the state, as
   nested sampling does. For coordinate-wise slice-within-Gibbs, use
   :func:`coordinate_slice`.

   .. rubric:: Examples

   A new slice sampling kernel can be initialized and used with the following
   code:

   .. code::

       slice_sampling = blackjax.slice_sampling(logdensity_fn)
       state = slice_sampling.init(position)
       new_state, info = slice_sampling.step(rng_key, state)

   :param logdensity_fn: Log-density of the distribution to sample from.
   :param proposal_generator: Proposal generator ``(rng_key, position, logdensity_fn) -> slice_fn``,
                              where ``slice_fn(t) -> (state, is_valid)``. Defaults to
                              ``direction_proposal()`` (isotropic unit directions).
   :param width: Initial bracket width along the direction (default 1.0).
   :param interval: Interval procedure :func:`doubling` (default) or :func:`stepping_out`,
                    passed as a callable.
   :param max_expansions: Caps on interval expansion and shrinkage.
   :param max_shrinkage: Caps on interval expansion and shrinkage.

   :rtype: A ``SamplingAlgorithm``.


.. py:function:: coordinate_slice(logdensity_fn: Callable, *, max_expansions: int = 10, initial_widths: float | blackjax.types.Array = 1.0, interval: Callable = doubling, coordinate_order: Callable = random_order, axis_proposal: Callable = coordinate_proposal, max_shrinkage: int = 100) -> blackjax.base.SamplingAlgorithm

   Coordinate-wise (slice-within-Gibbs) slice sampler.

   Updates each scalar coordinate's full conditional with a univariate slice,
   swept in the order given by ``coordinate_order``. The single-variable
   counterpart to the multivariate :func:`as_top_level_api`.

   :param logdensity_fn: Log-density of the distribution to sample from.
   :param max_expansions: Cap on interval expansions per coordinate (default 10).
   :param initial_widths: Scalar or per-coordinate initial bracket width(s) (default 1.0).
   :param interval: Interval procedure :func:`doubling` (default) or :func:`stepping_out`,
                    passed as a callable.
   :param coordinate_order: Choice function ``(rng_key, d) -> indices``, either :func:`random_order`
                            (default) or :func:`fixed_order`.
   :param axis_proposal: Per-axis proposal ``(rng_key, position, logdensity_fn, i) -> slice_fn``
                         (:func:`coordinate_proposal` by default). Override to gate a constraint
                         into ``is_valid``, as a custom ``proposal_generator`` does for
                         :func:`as_top_level_api`.
   :param max_shrinkage: Cap on shrinkage evaluations per coordinate.

   :rtype: A ``SamplingAlgorithm``.


