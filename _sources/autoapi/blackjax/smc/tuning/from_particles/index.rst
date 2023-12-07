:py:mod:`blackjax.smc.tuning.from_particles`
============================================

.. py:module:: blackjax.smc.tuning.from_particles

.. autoapi-nested-parse::

   strategies to tune the parameters of mcmc kernels
   used within SMC, based on particles.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.smc.tuning.from_particles.particles_stds
   blackjax.smc.tuning.from_particles.particles_means
   blackjax.smc.tuning.from_particles.particles_covariance_matrix
   blackjax.smc.tuning.from_particles.mass_matrix_from_particles



.. py:function:: particles_stds(particles)


.. py:function:: particles_means(particles)


.. py:function:: particles_covariance_matrix(particles)


.. py:function:: mass_matrix_from_particles(particles) -> blackjax.types.Array

   Implements tuning from section 3.1 from https://arxiv.org/pdf/1808.07730.pdf
   Computing a mass matrix to be used in HMC from particles.
   Given the particles covariance matrix, set all non-diagonal elements as zero,
   take the inverse, and keep the diagonal.

   :rtype: A mass Matrix


