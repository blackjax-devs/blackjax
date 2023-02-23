:py:mod:`blackjax.mcmc.metrics`
===============================

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



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.metrics.gaussian_euclidean



.. py:function:: gaussian_euclidean(inverse_mass_matrix: blackjax.types.Array) -> Tuple[Callable, EuclideanKineticEnergy, Callable]

   Hamiltonian dynamic on euclidean manifold with normally-distributed momentum :cite:p:`betancourt2013general`.

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


