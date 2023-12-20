:py:mod:`blackjax.mcmc.integrators`
===================================

.. py:module:: blackjax.mcmc.integrators

.. autoapi-nested-parse::

   Symplectic, time-reversible, integrators for Hamiltonian trajectories.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.integrators.implicit_midpoint



Attributes
~~~~~~~~~~

.. autoapisummary::

   blackjax.mcmc.integrators.velocity_verlet
   blackjax.mcmc.integrators.mclachlan
   blackjax.mcmc.integrators.yoshida
   blackjax.mcmc.integrators.isokinetic_leapfrog
   blackjax.mcmc.integrators.isokinetic_yoshida
   blackjax.mcmc.integrators.isokinetic_mclachlan


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

   

.. py:data:: isokinetic_leapfrog

   

.. py:data:: isokinetic_yoshida

   

.. py:data:: isokinetic_mclachlan

   

.. py:function:: implicit_midpoint(logdensity_fn: Callable, kinetic_energy_fn: blackjax.mcmc.metrics.KineticEnergy, *, solver: FixedPointSolver = solve_fixed_point_iteration, **solver_kwargs: Any) -> Integrator

   The implicit midpoint integrator with support for non-stationary kinetic energy

   This is an integrator based on :cite:t:`brofos2021evaluating`, which provides
   support for kinetic energies that depend on position. This integrator requires that
   the kinetic energy function takes two arguments: position and momentum.

   The ``solver`` parameter allows overloading of the fixed point solver. By default, a
   simple fixed point iteration is used, but more advanced solvers could be implemented
   in the future.


