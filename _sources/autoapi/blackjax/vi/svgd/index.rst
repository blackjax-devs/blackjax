:py:mod:`blackjax.vi.svgd`
==========================

.. py:module:: blackjax.vi.svgd


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.vi.svgd.init
   blackjax.vi.svgd.build_kernel
   blackjax.vi.svgd.rbf_kernel
   blackjax.vi.svgd.update_median_heuristic
   blackjax.vi.svgd.as_top_level_api



.. py:function:: init(initial_particles: blackjax.types.ArrayLikeTree, kernel_parameters: dict[str, Any], optimizer: optax.GradientTransformation) -> SVGDState

   Initializes Stein Variational Gradient Descent Algorithm.

   :param initial_particles: Initial set of particles to start the optimization
   :param kernel_paremeters: Arguments to the kernel function
   :param optimizer: Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol


.. py:function:: build_kernel(optimizer: optax.GradientTransformation)


.. py:function:: rbf_kernel(x, y, length_scale=1)


.. py:function:: update_median_heuristic(state: SVGDState) -> SVGDState

   Median heuristic for setting the bandwidth of RBF kernels.

   A reasonable middle-ground for choosing the `length_scale` of the RBF kernel
   is to pick the empirical median of the squared distance between particles.
   This strategy is called the median heuristic.


.. py:function:: as_top_level_api(grad_logdensity_fn: Callable, optimizer, kernel: Callable = rbf_kernel, update_kernel_parameters: Callable = update_median_heuristic)

   Implements the (basic) user interface for the svgd algorithm.

   :param grad_logdensity_fn: gradient, or an estimate, of the target log density function to samples approximately from
   :param optimizer: Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
   :param kernel: positive semi definite kernel
   :param update_kernel_parameters: function that updates the kernel parameters given the current state of the particles

   :rtype: A ``SamplingAlgorithm``.


