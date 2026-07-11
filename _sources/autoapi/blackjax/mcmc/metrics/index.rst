blackjax.mcmc.metrics
=====================

.. py:module:: blackjax.mcmc.metrics

.. autoapi-nested-parse::

   Metric space in which the Hamiltonian dynamic is embedded.

   An important particular case (and the most used in practice) of metric for the
   position space in the Euclidean metric. It is defined by a definite positive
   matrix :math:`M` with fixed value so that the kinetic energy of the hamiltonian
   dynamic is independent of the position and only depends on the momentum
   :math:`p` :cite:p:`betancourt2017geometric`.

   For a Newtonian hamiltonian dynamic the kinetic energy is given by:

   .. math::

       K(p) = \frac{1}{2} p^T M^{-1} p

   We can also generate a relativistic dynamic :cite:p:`lu2017relativistic`.



Classes
-------

.. autoapisummary::

   blackjax.mcmc.metrics.LowRankInverseMassMatrix


Functions
---------

.. autoapisummary::

   blackjax.mcmc.metrics.default_metric
   blackjax.mcmc.metrics.gaussian_euclidean
   blackjax.mcmc.metrics.gaussian_euclidean_low_rank
   blackjax.mcmc.metrics.gaussian_riemannian
   blackjax.mcmc.metrics.lbfgs_inverse_hessian_to_low_rank_metric


Module Contents
---------------

.. py:class:: LowRankInverseMassMatrix



   Pure-array description of a low-rank inverse mass matrix.

   The inverse mass matrix has the form

   .. math::

       M^{-1} = \operatorname{diag}(\sigma)
                \bigl(I + U(\Lambda - I)U^\top\bigr)
                \operatorname{diag}(\sigma)

   where :math:`\sigma \in \mathbb{R}^d_{>0}`, :math:`U \in \mathbb{R}^{d \times k}`
   has orthonormal columns and :math:`\Lambda = \operatorname{diag}(\lambda)`.

   This is the array-only payload produced by
   :func:`~blackjax.adaptation.low_rank_adaptation.window_adaptation_low_rank`.
   Unlike a fully-constructed :class:`Metric` (whose fields are Python closures
   that capture these arrays), this NamedTuple is a pure JAX pytree and can be
   safely transported across ``jax.vmap`` / ``jax.pmap`` boundaries.

   :func:`default_metric` expands this into a :class:`Metric` at the kernel
   call site via :func:`gaussian_euclidean_low_rank`.

   .. attribute:: sigma

      Shape ``(d,)``. Positive diagonal scaling.

   .. attribute:: U

      Shape ``(d, k)``. Matrix with orthonormal columns.

   .. attribute:: lam

      Shape ``(k,)``. Positive eigenvalues.


   .. py:attribute:: sigma
      :type:  blackjax.types.Array


   .. py:attribute:: U
      :type:  blackjax.types.Array


   .. py:attribute:: lam
      :type:  blackjax.types.Array


.. py:function:: default_metric(metric: MetricTypes) -> Metric

   Convert an input metric into a ``Metric`` object following sensible default rules.

   The metric can be specified in four different ways:

   - A ``Metric`` object that implements the full interface
   - A ``LowRankInverseMassMatrix`` NamedTuple holding ``(sigma, U, lam)``,
     which is expanded to a full :class:`Metric` via
     :func:`gaussian_euclidean_low_rank`. This is the form returned by
     :func:`~blackjax.adaptation.low_rank_adaptation.window_adaptation_low_rank`
     and is safe to transport across ``jax.vmap`` boundaries.
   - An ``Array`` which is assumed to specify the inverse mass matrix of a static
     metric
   - A function that takes a coordinate position and returns the mass matrix at that
     location

   :returns: * A ``Metric`` object with ``sample_momentum``, ``kinetic_energy``,
             * ``check_turning``, and ``scale`` fields.


.. py:function:: gaussian_euclidean(inverse_mass_matrix: blackjax.types.Array) -> Metric

   Hamiltonian dynamic on euclidean manifold with normally-distributed momentum
   :cite:p:`betancourt2013general`.

   The gaussian euclidean metric is a euclidean metric further characterized
   by setting the conditional probability density :math:`\pi(momentum|position)`
   to follow a standard gaussian distribution. A Newtonian hamiltonian
   dynamics is assumed.

   :param inverse_mass_matrix: One or two-dimensional array corresponding respectively to a diagonal
                               or dense mass matrix. The inverse mass matrix is multiplied to a
                               flattened version of the Pytree in which the chain position is stored
                               (the current value of the random variables). The order of the variables
                               should thus match JAX's tree flattening order, and more specifically
                               that of `ravel_pytree`.
                               In particular, JAX sorts dictionaries by key when flattening them. The
                               value of each variables will appear in the flattened Pytree following
                               the order given by `sort(keys)`.

   :returns: * *momentum_generator* -- A function that generates a value for the momentum at random.
             * *kinetic_energy* -- A function that returns the kinetic energy given the momentum.
             * *is_turning* -- A function that determines whether a trajectory is turning back on
               itself given the values of the momentum along the trajectory.


.. py:function:: gaussian_euclidean_low_rank(sigma: blackjax.types.Array, U: blackjax.types.Array, lam: blackjax.types.Array) -> Metric

   Euclidean metric with low-rank-modified mass matrix :cite:p:`seyboldt2026preconditioning`.

   The inverse mass matrix has the form

   .. math::

       M^{-1} = \operatorname{diag}(\sigma)
                \bigl(I + U(\Lambda - I)U^\top\bigr)
                \operatorname{diag}(\sigma)

   where :math:`\sigma \in \mathbb{R}^d_{>0}` is a diagonal scaling,
   :math:`U \in \mathbb{R}^{d \times k}` has orthonormal columns, and
   :math:`\Lambda = \operatorname{diag}(\lambda)` with :math:`\lambda > 0`.
   When :math:`\lambda = \mathbf{1}` the metric reduces to a diagonal metric
   with scale :math:`\sigma`.  All HMC operations are :math:`O(dk)`, making
   this efficient when :math:`k \ll d`.

   :param sigma: Shape ``(d,)``.  Positive diagonal scaling; plays the role of marginal
                 standard deviations.
   :param U: Shape ``(d, k)``.  Matrix with orthonormal columns spanning the
             low-rank correction subspace.
   :param lam: Shape ``(k,)``.  Positive eigenvalues for the low-rank correction.

   :rtype: A ``Metric`` object whose operations all run in :math:`O(dk)`.


.. py:function:: gaussian_riemannian(mass_matrix_fn: Callable) -> Metric

   Hamiltonian dynamic on Riemannian manifold with normally-distributed momentum.

   :param mass_matrix_fn: A callable that takes a position and returns the mass matrix at that
                          location (positive definite, one or two-dimensional array).

   :returns: * A ``Metric`` object with ``sample_momentum``, ``kinetic_energy``,
             * ``check_turning``, and ``scale`` fields.


.. py:function:: lbfgs_inverse_hessian_to_low_rank_metric(alpha: blackjax.types.Array, beta: blackjax.types.Array, gamma: blackjax.types.Array) -> LowRankInverseMassMatrix

   Convert an L-BFGS factored inverse-Hessian to a :class:`LowRankInverseMassMatrix`.

   The L-BFGS inverse Hessian is stored in the factored form

   .. math::

       H^{-1} = \operatorname{diag}(\alpha) + \beta \Gamma \beta^\top

   (formula II.1 / II.3 of :cite:p:`zhang2022pathfinder`).  This adapter
   rewrites it as a :class:`LowRankInverseMassMatrix` with

   .. math::

       M^{-1} = \operatorname{diag}(\sigma)
                \bigl(I + U(\Lambda - I)U^\top\bigr)
                \operatorname{diag}(\sigma),
       \quad \sigma = \sqrt{\alpha}

   via a compact :math:`O((2m)^3)` inner eigendecomposition that avoids the
   :math:`O(d^3)` full eigenproblem:

   1. Set :math:`D = \operatorname{diag}(\sigma)` and factor out to obtain
      :math:`H^{-1} = D(I + \tilde B \Gamma \tilde B^\top)D` with
      :math:`\tilde B = D^{-1}\beta \in \mathbb{R}^{d \times 2m}`.
   2. QR-decompose :math:`\tilde B = Q R` (thin QR, :math:`Q` orthonormal
      :math:`d \times r`, :math:`r = \min(d, 2m)`).
   3. The inner correction satisfies
      :math:`\tilde B \Gamma \tilde B^\top = Q (R \Gamma R^\top) Q^\top`,
      so its eigenvalues are those of the :math:`r \times r` matrix
      :math:`R \Gamma R^\top`.
   4. Eigendecompose :math:`R \Gamma R^\top = V \Lambda_c V^\top` (eigh,
      :math:`r \times r` only).
   5. Return :math:`U = QV` (orthonormal eigenvectors, shape :math:`d \times r`)
      and :math:`\lambda = 1 + \Lambda_c` (eigenvalues of :math:`I + \tilde B
      \Gamma \tilde B^\top`).

   **When to use this.**  Pass the ``(alpha, beta, gamma)`` triple produced by
   :func:`~blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_factors` to obtain a
   JAX-pytree-safe :class:`LowRankInverseMassMatrix` that can cross
   ``jax.vmap`` boundaries and feed any consumer that accepts the unified
   representation (HMC, MCLMC, etc.).

   .. note::
       This function is a **pure adapter** — it does not alter Pathfinder's
       internal sampling path (Phase R3 consumer migration).  In Phase R1 it
       ships as adapter + golden tests only.

   .. warning::
       **Positive-definiteness precondition.**  The triple ``(alpha, beta, gamma)``
       must yield a positive-definite dense form ``diag(alpha) + beta @ gamma @
       beta.T``; this is guaranteed when the triple comes from
       :func:`~blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_factors` under a
       Wolfe-condition line search.  Non-positive-definite inputs produce ``lam <=
       0`` *silently* here and surface as NaN at momentum sampling.  Additionally,
       at float32 near-singular metrics (condition number ≳ 1e7) can resolve the
       smallest eigenvalue with unreliable sign; for such inputs prefer float64
       factors.

   :param alpha: Shape ``(d,)``.  Positive diagonal of the inverse Hessian approximation.
   :param beta: Shape ``(d, 2m)``.  Left factor of the low-rank correction.  When
                ``m = 0`` (empty L-BFGS history) ``beta`` has shape ``(d, 0)`` and the
                adapter returns a pure diagonal metric.
   :param gamma: Shape ``(2m, 2m)``.  Symmetric inner factor of the low-rank correction.

   :returns: With ``sigma = sqrt(alpha)``, ``U`` the ``(d, r)`` orthonormal
             eigenvector matrix, and ``lam = 1 + eigenvalues(R Γ Rᵀ)``.
             Empty-history edge: ``U`` has shape ``(d, 0)`` and ``lam`` shape ``(0,)``,
             representing a pure diagonal metric with scale ``sigma``.
   :rtype: LowRankInverseMassMatrix

   .. seealso::

      :py:obj:`LowRankInverseMassMatrix`
          Target representation consumed by :func:`gaussian_euclidean_low_rank`.

      :py:obj:`gaussian_euclidean_low_rank`
          Full metric protocol built from the representation.


