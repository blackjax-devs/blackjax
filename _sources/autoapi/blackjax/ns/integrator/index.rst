blackjax.ns.integrator
======================

.. py:module:: blackjax.ns.integrator

.. autoapi-nested-parse::

   Evidence integration for Nested Sampling.

   This module provides utilities for tracking the evidence integral during
   a Nested Sampling run. The NSIntegrator accumulates statistics as the algorithm
   compresses the prior volume, computing the marginal likelihood (evidence),
   the running prior-volume estimate, and the live-point evidence contribution.



Classes
-------

.. autoapisummary::

   blackjax.ns.integrator.NSIntegrator


Functions
---------

.. autoapisummary::

   blackjax.ns.integrator.init_integrator
   blackjax.ns.integrator.update_integrator


Module Contents
---------------

.. py:class:: NSIntegrator



   Integrator for computing the evidence integral in Nested Sampling.

   This accumulates statistics over the course of a Nested Sampling run,
   computing the evidence (marginal likelihood) and related quantities
   from the history of dead particles. These are derived quantities that
   can be reconstructed from the dead particle history.

   .. attribute:: logX

      The log of the current prior volume estimate.

   .. attribute:: logZ

      The accumulated log evidence estimate from the "dead" points.

   .. attribute:: logZ_live

      The current estimate of the log evidence contribution from the live points.


   .. py:attribute:: logX
      :type:  blackjax.types.Array


   .. py:attribute:: logZ
      :type:  blackjax.types.Array


   .. py:attribute:: logZ_live
      :type:  blackjax.types.Array


.. py:function:: init_integrator(particle_state: blackjax.ns.base.StateWithLogLikelihood) -> NSIntegrator

   Initialize the evidence integrator from the initial live points.

   :param particle_state: The initial state containing the live particles.

   :returns: The initial integrator with logX=0, logZ=-inf, and logZ_live computed
             from the initial live points.
   :rtype: NSIntegrator


.. py:function:: update_integrator(integrator: NSIntegrator, particle_state: blackjax.ns.base.StateWithLogLikelihood, dead_particles: blackjax.ns.base.StateWithLogLikelihood) -> NSIntegrator

   Update the evidence integrator after a Nested Sampling step.

   :param integrator: The current integrator state.
   :param particle_state: The updated live state after the NS step.
   :param dead_particles: The particles that died in this step.

   :returns: The updated integrator. logX is the prior volume after all num_deleted
             deletions, logZ is the accumulated log evidence, and logZ_live is the
             log evidence contribution from the remaining live points.
   :rtype: NSIntegrator


