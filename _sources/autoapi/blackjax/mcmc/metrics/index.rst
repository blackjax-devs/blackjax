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



Functions
---------

.. autoapisummary::

   blackjax.mcmc.metrics.default_metric
   blackjax.mcmc.metrics.gaussian_euclidean
   blackjax.mcmc.metrics.gaussian_euclidean_low_rank
   blackjax.mcmc.metrics.gaussian_riemannian


Module Contents
---------------

.. py:function:: default_metric(metric: MetricTypes) -> Metric

   Convert an input metric into a ``Metric`` object following sensible default rules.

   The metric can be specified in three different ways:

   - A ``Metric`` object that implements the full interface
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

   Euclidean metric with low-rank-modified mass matrix :cite:p:`sountsov2025preconditioning`.

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


