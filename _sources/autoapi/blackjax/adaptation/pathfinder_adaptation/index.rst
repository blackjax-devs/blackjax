blackjax.adaptation.pathfinder_adaptation
=========================================

.. py:module:: blackjax.adaptation.pathfinder_adaptation

.. autoapi-nested-parse::

   Implementation of the Pathfinder warmup for the HMC family of sampling algorithms.



Classes
-------

.. autoapisummary::

   blackjax.adaptation.pathfinder_adaptation.PathfinderAdaptationState


Functions
---------

.. autoapisummary::

   blackjax.adaptation.pathfinder_adaptation.base
   blackjax.adaptation.pathfinder_adaptation.pathfinder_adaptation


Module Contents
---------------

.. py:class:: PathfinderAdaptationState



   .. py:attribute:: ss_state
      :type:  blackjax.adaptation.step_size.DualAveragingAdaptationState


   .. py:attribute:: step_size
      :type:  float


   .. py:attribute:: inverse_mass_matrix
      :type:  blackjax.types.Array


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


.. py:function:: pathfinder_adaptation(algorithm, logdensity_fn: Callable, *, num_chains: int = 1, n_paths: int | None = None, num_samples_per_path: int = 200, psis_imm_n_samples: int = 2000, imm_estimator: Literal['lbfgs_psis_mixture', 'psis_empirical'] = 'lbfgs_psis_mixture', initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, adaptation_info_fn: Callable = return_all_adapt_info, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC family.

   Supports single-chain (original behaviour) and multi-chain dispatch via
   `blackjax.vi.multipathfinder`.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to sample.
   :param num_chains: Number of independent chains to initialise.  When 1 (default) the
                      behaviour is identical to the original single-chain pathfinder
                      adaptation.  When > 1, ``num_chains`` chains are run in parallel with
                      per-chain init positions drawn from the PSIS mixture and a shared
                      dense ``(d, d)`` IMM.
   :param n_paths: Number of independent L-BFGS paths for the multi-path Pathfinder run.
                   Defaults to ``num_chains`` (one path per chain).  Ignored when both
                   ``num_chains == 1`` and ``n_paths`` is ``None`` (or explicitly 1) ---
                   the original single-path code path is used in that case.
   :param num_samples_per_path: Number of samples drawn per path to estimate ELBO and PSIS weights.
                                Only used when ``effective_n_paths >= 2``.  Default 200.
   :param psis_imm_n_samples: Number of PSIS-resampled draws used to estimate the post-PSIS
                              empirical covariance for the IMM.  Only used when
                              ``imm_estimator="psis_empirical"`` and ``effective_n_paths >= 2``.
                              Default 2000.
   :param imm_estimator: Method for deriving the inverse mass matrix when ``effective_n_paths >= 2``.

                         ``"lbfgs_psis_mixture"`` *(default)*
                             Analytic PSIS-weighted mixture covariance via the law of total
                             variance (within-component + between-component terms).  Returns a
                             dense ``(d, d)`` matrix.  When ``n_paths=1`` this equals the
                             single-path L-BFGS inverse Hessian exactly.

                         ``"psis_empirical"``
                             Empirical covariance estimated from ``psis_imm_n_samples`` draws
                             importance-resampled according to PSIS weights.  Returns a dense
                             ``(d, d)`` matrix.

                         Ignored for the single-chain single-path dispatch (``num_chains=1``,
                         ``effective_n_paths=1``).  Passing a non-default value in that case
                         raises a ``UserWarning``.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param adaptation_info_fn: Function to select the adaptation info returned. See return_all_adapt_info
                              and get_filter_adapt_info_fn in blackjax.adaptation.base.  By default all
                              information is saved - this can result in excessive memory usage if the
                              information is unused.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :returns: * *A function that returns the last chain state and a sampling kernel with the*
             * *tuned parameter values from an initial state.*

   .. rubric:: Notes

   **Dispatch table** --- the ``(num_chains, effective_n_paths)`` combination
   selects the internal code path:

   +-----------+--------------------+--------------------+---------------------------------------------+
   | num_chains| effective_n_paths  | imm_estimator      | Behaviour                                   |
   +===========+====================+====================+=============================================+
   | 1         | 1                  | (ignored)          | **Original** single-path Pathfinder + DA.   |
   |           |                    |                    | scalar step_size, ``(d, d)`` IMM.           |
   +-----------+--------------------+--------------------+---------------------------------------------+
   | 1         | >= 2               | lbfgs_psis_mixture | Multipathfinder; analytic PSIS-weighted     |
   |           |                    |                    | mixture covariance; ``(d, d)`` dense IMM.   |
   +-----------+--------------------+--------------------+---------------------------------------------+
   | 1         | >= 2               | psis_empirical     | Multipathfinder; empirical covariance from  |
   |           |                    |                    | psis_imm_n_samples resampled draws;         |
   |           |                    |                    | ``(d, d)`` dense IMM.                       |
   +-----------+--------------------+--------------------+---------------------------------------------+
   | >= 2      | 1                  | (ignored)          | Single-path Pathfinder; broadcast init x    |
   |           |                    |                    | chains; vmap DA. step_size (num_chains,).   |
   |           |                    |                    | ``(d, d)`` IMM from L-BFGS inverse Hessian. |
   +-----------+--------------------+--------------------+---------------------------------------------+
   | >= 2      | >= 2               | lbfgs_psis_mixture | **Paper-canonical** (Zhang et al. 2022).    |
   |           |                    |                    | Multipathfinder -> PSIS init -> analytic    |
   |           |                    |                    | mixture covariance -> vmap DA.              |
   +-----------+--------------------+--------------------+---------------------------------------------+
   | >= 2      | >= 2               | psis_empirical     | Multipathfinder -> PSIS init -> empirical   |
   |           |                    |                    | covariance -> vmap DA.                      |
   +-----------+--------------------+--------------------+---------------------------------------------+

   **Return contract:**

   * ``num_chains == 1``: identical to the pre-change API.
     ``parameters["step_size"]`` is a scalar;
     ``parameters["inverse_mass_matrix"]`` is uniformly ``(d, d)``.
   * ``num_chains > 1``: ``parameters["step_size"]`` is ``(num_chains,)``;
     ``parameters["inverse_mass_matrix"]`` is the single shared ``(d, d)`` IMM
     (not broadcast --- the caller broadcasts if needed).
   * When ``effective_n_paths >= 2``, ``parameters`` also includes
     ``"_pathfinder_psis_pareto_k"`` (scalar) for downstream diagnostics.

   **PSIS-weighted mixture covariance math** (``imm_estimator="lbfgs_psis_mixture"``):

   Multipathfinder produces a mixture of Laplace approximations
   ``p(theta) approx sum_i w_i * N(mu_i, Sigma_i)`` where ``mu_i`` is path ``i``'s
   L-BFGS optimum position and ``Sigma_i`` is its L-BFGS inverse-Hessian estimate.
   Weights ``w_i`` are aggregate (per-path sum) PSIS weights.

   By the law of total variance:

   .. math::

       \Sigma_{\text{mix}} = \sum_i w_i \Sigma_i
           + \sum_i w_i (\mu_i - \bar\mu)(\mu_i - \bar\mu)^T

   where :math:`\bar\mu = \sum_i w_i \mu_i`.


