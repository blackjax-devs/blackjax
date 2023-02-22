:py:mod:`blackjax.optimizers.lbfgs`
===================================

.. py:module:: blackjax.optimizers.lbfgs


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.optimizers.lbfgs.LBFGSHistory



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.optimizers.lbfgs.minimize_lbfgs
   blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_factors
   blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_formula_1
   blackjax.optimizers.lbfgs.lbfgs_inverse_hessian_formula_2
   blackjax.optimizers.lbfgs.bfgs_sample



.. py:class:: LBFGSHistory



   Container for the optimization path of a L-BFGS run

   x
       History of positions
   f
       History of objective values
   g
       History of gradient values
   alpha
       History of the diagonal elements of the inverse Hessian approximation.
   update_mask:
       The indicator of whether the updates of position and gradient are
       included in the inverse-Hessian approximation or not.
       (Xi in the paper)


   .. py:attribute:: x
      :type: blackjax.types.Array

      

   .. py:attribute:: f
      :type: blackjax.types.Array

      

   .. py:attribute:: g
      :type: blackjax.types.Array

      

   .. py:attribute:: alpha
      :type: blackjax.types.Array

      

   .. py:attribute:: update_mask
      :type: blackjax.types.Array

      


.. py:function:: minimize_lbfgs(fun: Callable, x0: blackjax.types.PyTree, maxiter: int = 30, maxcor: float = 10, gtol: float = 1e-08, ftol: float = 1e-05, maxls: int = 1000) -> Tuple[jaxopt.base.OptStep, LBFGSHistory]

   Minimize a function using L-BFGS

   :param fun: function of the form f(x) where x is a pytree and returns a real scalar.
               The function should be composed of operations with vjp defined.
   :param x0: initial guess
   :param maxiter: maximum number of iterations
   :param maxcor: maximum number of metric corrections ("history size")
   :param ftol: terminates the minimization when `(f_k - f_{k+1}) < ftol`
   :param gtol: terminates the minimization when `|g_k|_norm < gtol`
   :param maxls: maximum number of line search steps (per iteration)

   :rtype: Optimization results and optimization path


.. py:function:: lbfgs_inverse_hessian_factors(S, Z, alpha)

   Calculates factors for inverse hessian factored representation.
   It implements formula II.2 of:

   Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782



.. py:function:: lbfgs_inverse_hessian_formula_1(alpha, beta, gamma)

   Calculates inverse hessian from factors as in formula II.1 of:

   Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782



.. py:function:: lbfgs_inverse_hessian_formula_2(alpha, beta, gamma)

   Calculates inverse hessian from factors as in formula II.3 of:

   Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782



.. py:function:: bfgs_sample(rng_key, num_samples, position, grad_position, alpha, beta, gamma)

   Draws approximate samples of target distribution.
   It implements Algorithm 4 in:

   Pathfinder: Parallel quasi-newton variational inference, Lu Zhang et al., arXiv:2108.03782



