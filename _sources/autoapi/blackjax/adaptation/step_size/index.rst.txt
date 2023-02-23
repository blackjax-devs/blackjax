:py:mod:`blackjax.adaptation.step_size`
=======================================

.. py:module:: blackjax.adaptation.step_size

.. autoapi-nested-parse::

   Step size adaptation



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.step_size.DualAveragingAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.step_size.dual_averaging_adaptation
   blackjax.adaptation.step_size.find_reasonable_step_size



.. py:class:: DualAveragingAdaptationState



   State carried through the dual averaging procedure.

   log_step_size
       The logarithm of the current value of the step size.
   log_step_size_avg
       The time-weighted average of the values that the logarithm of the step
       size has taken so far.
   step
       The current iteration step.
   avg_err
       The time average of the value of the quantity :math:`H_t`, the
       difference between the target acceptance rate and the current
       acceptance rate.
   mu
       Arbitrary point the values of log_step_size are shrunk towards. Chose
       to be :math:`\log(10 \epsilon_0)` where :math:`\epsilon_0` is chosen
       in this context to be the step size given by the
       `find_reasonable_step_size` procedure.


   .. py:attribute:: log_step_size
      :type: float

      

   .. py:attribute:: log_step_size_avg
      :type: float

      

   .. py:attribute:: step
      :type: int

      

   .. py:attribute:: avg_error
      :type: float

      

   .. py:attribute:: mu
      :type: float

      


.. py:function:: dual_averaging_adaptation(target: float, t0: int = 10, gamma: float = 0.05, kappa: float = 0.75) -> Tuple[Callable, Callable, Callable]

   Tune the step size in order to achieve a desired target acceptance rate.

   Let us note :math:`\epsilon` the current step size, :math:`\alpha_t` the
   metropolis acceptance rate at time :math:`t` and :math:`\delta` the desired
   aceptance rate. We define:

   .. math:
       H_t = \delta - \alpha_t

   the error at time t. We would like to find a procedure that adapts the
   value of :math:`\epsilon` such that :math:`h(x) =\mathbb{E}\left[H_t|\epsilon\right] = 0`

   Following :cite:p:`nesterov2009primal`, the authors of :cite:p:`hoffman2014no` proposed the following update scheme. If
   we note :math:`x = \log \epsilon` we follow:

   .. math:
       x_{t+1} \LongLeftArrow \mu - \frac{\sqrt{t}}{\gamma} \frac{1}{t+t_0} \sum_{i=1}^t H_i
       \overline{x}_{t+1} \LongLeftArrow x_{t+1}\, t^{-\kappa}  + \left(1-t^\kappa\right)\overline{x}_t

   :math:`\overline{x}_{t}` is guaranteed to converge to a value such that
   :math:`h(\overline{x}_t)` converges to 0, i.e. the Metropolis acceptance
   rate converges to the desired rate.

   See reference :cite:p:`hoffman2014no` (section 3.2.1) for a detailed discussion.

   :param t0: Free parameter that stabilizes the initial iterations of the algorithm.
              Large values may slow down convergence. Introduced in :cite:p:`hoffman2014no` with a default
              value of 10.
   :type t0: float >= 0
   :param gamma: Controls the speed of convergence of the scheme. The authors of :cite:p:`hoffman2014no` recommend
                 a value of 0.05.
   :param kappa: Controls the weights of past steps in the current update. The scheme will
                 quickly forget earlier step for a small value of `kappa`. Introduced
                 in :cite:p:`hoffman2014no`, with a recommended value of .75
   :type kappa: float in ]0.5, 1]
   :param target: Target acceptance rate.

   :returns: * *init* -- A function that initializes the state of the dual averaging scheme.
             * *update* -- A function that updates the state of the dual averaging scheme.


.. py:function:: find_reasonable_step_size(rng_key: blackjax.types.PRNGKey, kernel_generator: Callable[[float], Callable], reference_state: blackjax.mcmc.hmc.HMCState, initial_step_size: float, target_accept: float = 0.65) -> float

   Find a reasonable initial step size during warmup.

   While the dual averaging scheme is guaranteed to converge to a reasonable
   value for the step size starting from any value, choosing a good first
   value can speed up the convergence. This heuristics doubles and halves the
   step size until the acceptance probability of the HMC proposal crosses the
   target value :cite:p:`hoffman2014no`.

   :param rng_key: Key used by JAX's random number generator.
   :param kernel_generator: A function that takes a step size as an input and returns the corresponding
                            sampling kernel.
   :param reference_hmc_state: The location (HMC state) where this first step size must be found. This function
                               never advances the chain.
   :param inverse_mass_matrix: The inverse mass matrix relative to which the step size must be found.
   :param initial_step_size: The first step size used to start the search.
   :param target_accept: Once that value of the metropolis acceptance probability is reached we
                         estimate that we have found a "reasonable" first step size.

   :returns: A reasonable first value for the step size.
   :rtype: float


