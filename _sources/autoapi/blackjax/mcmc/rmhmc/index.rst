:py:mod:`blackjax.mcmc.rmhmc`
=============================

.. py:module:: blackjax.mcmc.rmhmc


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.mcmc.rmhmc.rmhmc




Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.rmhmc.init
   blackjax.mcmc.rmhmc.build_kernel


.. py:data:: init

   

.. py:data:: build_kernel

   

.. py:class:: rmhmc


   A Riemannian Manifold Hamiltonian Monte Carlo kernel

   Of note, this kernel is simply an alias of the ``hmc`` kernel with a
   different choice of default integrator (``implicit_midpoint`` instead of
   ``velocity_verlet``) since RMHMC is typically used for Hamiltonian systems
   that are not separable.

   :param logdensity_fn: The log-density function we wish to draw samples from.
   :param step_size: The value to use for the step size in the symplectic integrator.
   :param mass_matrix: A function which computes the mass matrix (not inverse) at a given
                       position when drawing a value for the momentum and computing the kinetic
                       energy. In practice, this argument will be passed to the
                       ``metrics.default_metric`` function so it supports all the options
                       discussed there.
   :param num_integration_steps: The number of steps we take with the symplectic integrator at each
                                 sample step before returning a sample.
   :param divergence_threshold: The absolute value of the difference in energy between two states above
                                which we say that the transition is divergent. The default value is
                                commonly found in other libraries, and yet is arbitrary.
   :param integrator: (algorithm parameter) The symplectic integrator to use to integrate the
                      trajectory.

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


