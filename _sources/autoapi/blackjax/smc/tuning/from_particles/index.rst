blackjax.smc.tuning.from_particles
==================================

.. py:module:: blackjax.smc.tuning.from_particles

.. autoapi-nested-parse::

   static (all particles get the same value) strategies to tune the parameters of mcmc kernels
   used within SMC, based on particles.



Functions
---------

.. autoapisummary::

   blackjax.smc.tuning.from_particles.particles_stds
   blackjax.smc.tuning.from_particles.particles_means
   blackjax.smc.tuning.from_particles.particles_covariance_matrix
   blackjax.smc.tuning.from_particles.inverse_mass_matrix_from_particles


Module Contents
---------------

.. py:function:: particles_stds(particles)

.. py:function:: particles_means(particles)

.. py:function:: particles_covariance_matrix(particles)

.. py:function:: inverse_mass_matrix_from_particles(particles) -> blackjax.types.Array

   Implements tuning from section 3.1 from https://arxiv.org/pdf/1808.07730.pdf
   Computing an inverse mass matrix to be used in HMC from particles.

   :rtype: An inverse mass matrix


