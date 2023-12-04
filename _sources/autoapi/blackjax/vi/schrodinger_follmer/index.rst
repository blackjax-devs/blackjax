:py:mod:`blackjax.vi.schrodinger_follmer`
=========================================

.. py:module:: blackjax.vi.schrodinger_follmer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.vi.schrodinger_follmer.SchrodingerFollmerState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.vi.schrodinger_follmer.init
   blackjax.vi.schrodinger_follmer.step
   blackjax.vi.schrodinger_follmer.sample



.. py:class:: SchrodingerFollmerState




   State of the Schrödinger-Föllmer algorithm.

   The Schrödinger-Föllmer algorithm gets samples from the target distribution by
   approximating the target distribution as the terminal value of a stochastic differential
   equation (SDE) with a drift term that is evaluated under the running samples.

   position:
       position of the sample
   time:
       Current integration time of the SDE

   .. py:attribute:: position
      :type: blackjax.types.ArrayLikeTree

      

   .. py:attribute:: time
      :type: jax.typing.ArrayLike

      


.. py:function:: init(example_position: blackjax.types.ArrayLikeTree) -> SchrodingerFollmerState


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: SchrodingerFollmerState, logdensity_fn: Callable, step_size: float, n_samples: int) -> Tuple[SchrodingerFollmerState, SchrodingerFollmerInfo]

   Runs one step of the Schrödinger-Föllmer algorithm. As per the paper, we only allow for Euler-Maruyama integration.
   It is likely possible to generalize this to other integration schemes but is not considered in the original work
   and we therefore do not consider it here.

   Note that we use the version with Stein's lemma as computing the gradient of the *density* is typically unstable.

   :param rng_key: PRNG key
   :param state: Current state of the algorithm
   :param logdensity_fn: Log-density of the target distribution
   :param step_size: Step size of the integration scheme
   :param n_samples: Number of samples to use to approximate the drift term


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, initial_state: SchrodingerFollmerState, log_density_fn: Callable, n_steps: int, n_inner_samples, n_samples: int = 1)

   Samples from the target distribution using the Schrödinger-Föllmer algorithm.

   :param rng_key: PRNG key
   :param initial_state: Current state of the algorithm
   :param log_density_fn: Log-density of the target distribution
   :param n_steps: Number of steps to run the algorithm for
   :param n_inner_samples: Number of samples to use to approximate the drift term
   :param n_samples: Number of samples to draw


