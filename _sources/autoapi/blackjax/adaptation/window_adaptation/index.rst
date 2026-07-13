blackjax.adaptation.window_adaptation
=====================================

.. py:module:: blackjax.adaptation.window_adaptation

.. autoapi-nested-parse::

   Implementation of the Stan warmup for the HMC family of sampling algorithms.

   The public surface of this module is unchanged.  Internally, :func:`window_adaptation`
   is now a thin compatibility shim over :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`;
   :func:`build_schedule` is defined in :mod:`blackjax.adaptation.staged_adaptation` and
   re-exported here for backward compatibility.

   :data:`WindowAdaptationState` is an alias for
   :class:`~blackjax.adaptation.staged_adaptation.StagedAdaptationState`; both names
   refer to the same class object so ``isinstance`` checks using either name continue
   to work without modification.

   The :func:`base` function is retained at its released API for downstream code that
   calls it directly.  It is not exercised by the :func:`window_adaptation` shim (which
   delegates to :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`).
   Fisher-diagonal adaptation is accessible via
   ``staged_adaptation(metric="fisher_diag")`` only.



Attributes
----------

.. autoapisummary::

   blackjax.adaptation.window_adaptation.WindowAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.window_adaptation.base
   blackjax.adaptation.window_adaptation.window_adaptation


Module Contents
---------------

.. py:data:: WindowAdaptationState

.. py:function:: base(is_mass_matrix_diagonal: bool, target_acceptance_rate: float = 0.8, initial_inverse_mass_matrix: blackjax.types.Array | None = None, imm_shrinkage_to_previous: float = 0.0) -> tuple[Callable, Callable, Callable]

   Warmup scheme for sampling procedures based on euclidean manifold HMC.
   The schedule and algorithms used match Stan's :cite:p:`stan_hmc_param` as closely as possible.

   Unlike several other libraries, we separate the warmup and sampling phases
   explicitly. This ensure a better modularity; a change in the warmup does
   not affect the sampling. It also allows users to run their own warmup
   should they want to.
   We also decouple generating a new sample with the mcmc algorithm and
   updating the values of the parameters.

   Stan's warmup consists in the three following phases:

   1. A fast adaptation window where only the step size is adapted using
   Nesterov's dual averaging scheme to match a target acceptance rate.
   2. A succession of slow adapation windows (where the size of a window is
   double that of the previous window) where both the mass matrix and the step
   size are adapted. The mass matrix is recomputed at the end of each window;
   the step size is re-initialized to a "reasonable" value.
   3. A last fast adaptation window where only the step size is adapted.

   Schematically:

   +---------+---+------+------------+------------------------+------+
   |  fast   | s | slow |   slow     |        slow            | fast |
   +---------+---+------+------------+------------------------+------+
   |1        |2  |3     |3           |3                       |3     |
   +---------+---+------+------------+------------------------+------+

   Step (1) consists in find a "reasonable" first step size that is used to
   initialize the dual averaging scheme. In (2) we initialize the mass matrix
   to the matrix. In (3) we compute the mass matrix to use in the kernel and
   re-initialize the mass matrix adaptation. The step size is still adapated
   in slow adaptation windows, and is not re-initialized between windows.

   :param is_mass_matrix_diagonal: Create and adapt a diagonal mass matrix if True, a dense matrix
                                   otherwise.
   :param target_acceptance_rate: The target acceptance rate for the step size adaptation.
   :param initial_inverse_mass_matrix: Optional seed value for the inverse mass matrix passed through to
                                       ``mass_matrix_adaptation``.  ``None`` (default) uses the standard
                                       identity initialisation.
   :param imm_shrinkage_to_previous: Pseudo-count controlling shrinkage of the IMM toward the previous
                                     window's IMM. Default 0.0 gives the current Stan behavior. Passed
                                     through to ``mass_matrix_adaptation``.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.
             * *final* -- Function that returns the step size and mass matrix given a warmup
               state.


.. py:function:: window_adaptation(algorithm, logdensity_fn: Callable, is_mass_matrix_diagonal: bool = True, initial_inverse_mass_matrix: blackjax.types.Array | None = None, imm_shrinkage_to_previous: float = 0.0, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, adaptation_info_fn: Callable = return_all_adapt_info, integrator=mcmc.integrators.velocity_verlet, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC fmaily. See Blackjax.hmc_family

   Algorithms in the HMC family on a euclidean manifold depend on the value of
   at least two parameters: the step size, related to the trajectory
   integrator, and the mass matrix, linked to the euclidean metric.

   Good tuning is very important, especially for algorithms like NUTS which can
   be extremely inefficient with the wrong parameter values. This function
   provides a general-purpose algorithm to tune the values of these parameters.
   Originally based on Stan's window adaptation, the algorithm has evolved to
   improve performance and quality.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to
                         sample.
   :param is_mass_matrix_diagonal: Whether we should adapt a diagonal mass matrix.
   :param initial_inverse_mass_matrix: Optional seed value for the inverse mass matrix used at the start of
                                       warmup.  When ``None`` (default) the standard identity initialisation
                                       is used (``ones(d)`` for diagonal, ``identity(d)`` for dense).  When
                                       provided the array seeds the first window's step-size adaptation with a
                                       better geometric hint; the Welford algorithm still starts from scratch
                                       so the seed is gradually overwritten by the empirical covariance.

                                       Shape must be consistent with ``is_mass_matrix_diagonal``:

                                       * diagonal (``is_mass_matrix_diagonal=True``): 1-D array of shape
                                         ``(d,)`` where ``d`` is the number of model parameters.
                                       * dense (``is_mass_matrix_diagonal=False``): 2-D square array of shape
                                         ``(d, d)``.

                                       A ``ValueError`` is raised at construction time (before any JIT
                                       tracing) if the shape is inconsistent.
   :param imm_shrinkage_to_previous: Bayesian pseudo-count controlling shrinkage of the per-window
                                     adapted inverse mass matrix toward the *previous* window's IMM, in
                                     addition to the existing Stan-style shrinkage toward
                                     ``1e-3 · I`` (pseudo-count 5). Default ``0.0`` reproduces Stan's
                                     behavior exactly: each window's Welford estimate replaces the
                                     previous IMM (no persistence). A positive value blends a fraction
                                     ``k_prev / (count + 5 + k_prev)`` of the previous IMM into the
                                     new one, where ``count`` is the number of samples in the window
                                     and ``k_prev`` is this argument.

                                     Useful when ``initial_inverse_mass_matrix`` carries high-confidence
                                     information (e.g., from a converged pre-warmup Pathfinder fit) that
                                     should persist beyond window 1's reset. Practical band for typical
                                     Stan window sizes (25–500): ``5 ≤ k_prev ≤ 50`` gives mild-to-
                                     moderate persistence; ``k_prev ≈ window_size`` gives balanced 50/50
                                     weight between the previous IMM and the new window's data;
                                     ``k_prev >> window_size`` effectively freezes the IMM at
                                     ``initial_inverse_mass_matrix`` (anti-pattern unless the seed is
                                     truly known-correct). See ``mass_matrix_adaptation`` for the full
                                     precision-weighted-average formula.

                                     Validated at construction time — negative values raise
                                     ``ValueError`` before any JIT tracing.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param adaptation_info_fn: Function to select the adaptation info returned. See return_all_adapt_info
                              and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
                              information is saved - this can result in excessive memory usage if the
                              information is unused.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :rtype: A function that runs the adaptation and returns an `AdaptationResult` object.

   .. rubric:: Notes

   This function is a thin compatibility shim over
   :func:`~blackjax.adaptation.staged_adaptation.staged_adaptation`.  The
   public interface and return type are frozen; no breaking changes will be
   made in this module.

   Wrap ``warmup.run(...)`` in :func:`blackjax.progress_bar` to display a
   progress bar, e.g. ``with blackjax.progress_bar(): warmup.run(...)``.


