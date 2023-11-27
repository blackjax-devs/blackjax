:py:mod:`blackjax.mcmc.integrators`
===================================

.. py:module:: blackjax.mcmc.integrators

.. autoapi-nested-parse::

   Symplectic, time-reversible, integrators for Hamiltonian trajectories.



Module Contents
---------------

.. py:data:: velocity_verlet

   Two-stage palindromic symplectic integrator derived in :cite:p:`blanes2014numerical`.

   The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
   determine both the bound on the integration error and the stability of the
   method with respect to the value of `step_size`. The values used here are
   the ones derived in :cite:p:`mclachlan1995numerical`; note that :cite:p:`blanes2014numerical`
   is more focused on stability and derives different values.

   Also known as the minimal norm integrator.

.. py:data:: mclachlan

   Three stages palindromic symplectic integrator derived in :cite:p:`mclachlan1995numerical`

   The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
   the parameters determine both the bound on the integration error and the
   stability of the method with respect to the value of `step_size`. The
   values used here are the ones derived in :cite:p:`mclachlan1995numerical` which
   guarantees a stability interval length approximately equal to 4.67.

.. py:data:: yoshida

   

.. py:data:: noneuclidean_leapfrog

   

.. py:data:: noneuclidean_mclachlan

   

.. py:data:: noneuclidean_yoshida

   

