:py:mod:`blackjax.adaptation.window_adaptation`
===============================================

.. py:module:: blackjax.adaptation.window_adaptation

.. autoapi-nested-parse::

   Implementation of the Stan warmup for the HMC family of sampling algorithms.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.adaptation.window_adaptation.WindowAdaptationState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.adaptation.window_adaptation.base
   blackjax.adaptation.window_adaptation.window_adaptation
   blackjax.adaptation.window_adaptation.build_schedule



.. py:class:: WindowAdaptationState




   .. py:attribute:: ss_state
      :type: blackjax.adaptation.step_size.DualAveragingAdaptationState

      

   .. py:attribute:: imm_state
      :type: blackjax.adaptation.mass_matrix.MassMatrixAdaptationState

      

   .. py:attribute:: step_size
      :type: float

      

   .. py:attribute:: inverse_mass_matrix
      :type: blackjax.types.Array

      


.. py:function:: base(is_mass_matrix_diagonal: bool, target_acceptance_rate: float = 0.8) -> tuple[Callable, Callable, Callable]

   Warmup scheme for sampling procedures based on euclidean manifold HMC.
   The schedule and algorithms used match Stan's :cite:p:`stan_hmc_param` as closely as possible.

   Unlike several other libraries, we separate the warmup and sampling phases
   explicitly. This ensure a better modularity; a change in the warmup does
   not affect the sampling. It also allows users to run their own warmup
   should they want to.
   We also decouple generating a new sample with the mcmc algorithm and
   updating the values of the parameters.

   Stan's warmup consists in the three following phases:

   1. A fast adaptation window where only the step size is adapted using
   Nesterov's dual averaging scheme to match a target acceptance rate.
   2. A succession of slow adapation windows (where the size of a window is
   double that of the previous window) where both the mass matrix and the step
   size are adapted. The mass matrix is recomputed at the end of each window;
   the step size is re-initialized to a "reasonable" value.
   3. A last fast adaptation window where only the step size is adapted.

   Schematically:

   +---------+---+------+------------+------------------------+------+
   |  fast   | s | slow |   slow     |        slow            | fast |
   +---------+---+------+------------+------------------------+------+
   |1        |2  |3     |3           |3                       |3     |
   +---------+---+------+------------+------------------------+------+

   Step (1) consists in find a "reasonable" first step size that is used to
   initialize the dual averaging scheme. In (2) we initialize the mass matrix
   to the matrix. In (3) we compute the mass matrix to use in the kernel and
   re-initialize the mass matrix adaptation. The step size is still adapated
   in slow adaptation windows, and is not re-initialized between windows.

   :param is_mass_matrix_diagonal: Create and adapt a diagonal mass matrix if True, a dense matrix
                                   otherwise.
   :param target_acceptance_rate: The target acceptance rate for the step size adaptation.

   :returns: * *init* -- Function that initializes the warmup.
             * *update* -- Function that moves the warmup one step.
             * *final* -- Function that returns the step size and mass matrix given a warmup
               state.


.. py:function:: window_adaptation(algorithm: Union[blackjax.mcmc.hmc.hmc, blackjax.mcmc.nuts.nuts], logdensity_fn: Callable, is_mass_matrix_diagonal: bool = True, initial_step_size: float = 1.0, target_acceptance_rate: float = 0.8, progress_bar: bool = False, **extra_parameters) -> blackjax.base.AdaptationAlgorithm

   Adapt the value of the inverse mass matrix and step size parameters of
   algorithms in the HMC fmaily.

   Algorithms in the HMC family on a euclidean manifold depend on the value of
   at least two parameters: the step size, related to the trajectory
   integrator, and the mass matrix, linked to the euclidean metric.

   Good tuning is very important, especially for algorithms like NUTS which can
   be extremely inefficient with the wrong parameter values. This function
   provides a general-purpose algorithm to tune the values of these parameters.
   Originally based on Stan's window adaptation, the algorithm has evolved to
   improve performance and quality.

   :param algorithm: The algorithm whose parameters are being tuned.
   :param logdensity_fn: The log density probability density function from which we wish to
                         sample.
   :param is_mass_matrix_diagonal: Whether we should adapt a diagonal mass matrix.
   :param initial_step_size: The initial step size used in the algorithm.
   :param target_acceptance_rate: The acceptance rate that we target during step size adaptation.
   :param progress_bar: Whether we should display a progress bar.
   :param \*\*extra_parameters: The extra parameters to pass to the algorithm, e.g. the number of
                                integration steps for HMC.

   :rtype: A function that runs the adaptation and returns an `AdaptationResult` object.


.. py:function:: build_schedule(num_steps: int, initial_buffer_size: int = 75, final_buffer_size: int = 50, first_window_size: int = 25) -> list[tuple[int, bool]]

   Return the schedule for Stan's warmup.

   The schedule below is intended to be as close as possible to Stan's :cite:p:`stan_hmc_param`.
   The warmup period is split into three stages:

   1. An initial fast interval to reach the typical set. Only the step size is
   adapted in this window.
   2. "Slow" parameters that require global information (typically covariance)
   are estimated in a series of expanding intervals with no memory; the step
   size is re-initialized at the end of each window. Each window is twice the
   size of the preceding window.
   3. A final fast interval during which the step size is adapted using the
   computed mass matrix.

   Schematically:

   ```
   +---------+---+------+------------+------------------------+------+
   |  fast   | s | slow |   slow     |        slow            | fast |
   +---------+---+------+------------+------------------------+------+
   ```

   The distinction slow/fast comes from the speed at which the algorithms
   converge to a stable value; in the common case, estimation of covariance
   requires more steps than dual averaging to give an accurate value. See :cite:p:`stan_hmc_param`
   for a more detailed explanation.

   Fast intervals are given the label 0 and slow intervals the label 1.

   :param num_steps: The number of warmup steps to perform.
   :type num_steps: int
   :param initial_buffer: The width of the initial fast adaptation interval.
   :type initial_buffer: int
   :param first_window_size: The width of the first slow adaptation interval.
   :type first_window_size: int
   :param final_buffer_size: The width of the final fast adaptation interval.
   :type final_buffer_size: int

   :rtype: A list of tuples (window_label, is_middle_window_end).


