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

   blackjax.mcmc.integrators.velocity_verlet
   blackjax.mcmc.integrators.mclachlan
   blackjax.mcmc.integrators.yoshida



.. py:function:: velocity_verlet(logdensity_fn: Callable, kinetic_energy_fn: blackjax.mcmc.metrics.EuclideanKineticEnergy) -> EuclideanIntegrator

   The velocity Verlet (or Verlet-Störmer) integrator.

   The velocity Verlet is a two-stage palindromic integrator :cite:p:`bou2018geometric` of the form
   (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
   the step size that range between 0 and 2 (when the mass matrix is the
   identity).

   While the position (a1 = 0.5) and velocity Verlet are the most commonly used
   in samplers, it is known in the numerical computation literature that the value
   $a1 pprox 0.1932$ leads to a lower integration error :cite:p:`mclachlan1995numerical,schlick2010molecular`. The authors of :cite:p:`bou2018geometric`
   show that the value $a1 pprox 0.21132$ leads to an even higher step acceptance
   rate, up to 3 times higher than with the standard position verlet (p.22, Fig.4).

   By choosing the velocity verlet we avoid two computations of the gradient
   of the kinetic energy. We are trading accuracy in exchange, and it is not
   clear whether this is the right tradeoff.



.. py:function:: mclachlan(logdensity_fn: Callable, kinetic_energy_fn: Callable) -> EuclideanIntegrator

   Two-stage palindromic symplectic integrator derived in :cite:p:`blanes2014numerical`.

   The integrator is of the form (b1, a1, b2, a1, b1). The choice of the parameters
   determine both the bound on the integration error and the stability of the
   method with respect to the value of `step_size`. The values used here are
   the ones derived in :cite:p:`mclachlan1995numerical`; note that :cite:p:`blanes2014numerical` is more focused on stability
   and derives different values.



.. py:function:: yoshida(logdensity_fn: Callable, kinetic_energy_fn: Callable) -> EuclideanIntegrator

   Three stages palindromic symplectic integrator derived in :cite:p:`mclachlan1995numerical`

   The integrator is of the form (b1, a1, b2, a2, b2, a1, b1). The choice of
   the parameters determine both the bound on the integration error and the
   stability of the method with respect to the value of `step_size`. The
   values used here are the ones derived in :cite:p:`mclachlan1995numerical` which guarantees a stability
   interval length approximately equal to 4.67.



