blackjax.vi.svgd
================

.. py:module:: blackjax.vi.svgd


Functions
---------

.. autoapisummary::

   blackjax.vi.svgd.init
   blackjax.vi.svgd.build_kernel
   blackjax.vi.svgd.rbf_kernel
   blackjax.vi.svgd.update_median_heuristic
   blackjax.vi.svgd.as_top_level_api


Module Contents
---------------

.. py:function:: init(initial_particles: blackjax.types.ArrayLikeTree, kernel_parameters: dict[str, Any], optimizer: optax.GradientTransformation) -> SVGDState

   Initializes Stein Variational Gradient Descent Algorithm.

   :param initial_particles: Initial set of particles to start the optimization.
   :param kernel_parameters: Arguments to the kernel function.
   :param optimizer: Optax compatible optimizer, which conforms to the ``optax.GradientTransformation`` protocol.

   :rtype: Initial SVGDState with the given particles, kernel parameters, and optimizer state.


.. py:function:: build_kernel(optimizer: optax.GradientTransformation)

   Build a SVGD kernel.

   :param optimizer: Optax optimizer used to apply the functional gradient update.

   :returns: * A ``kernel(state, grad_logdensity_fn, kernel, \*\*grad_params) -> SVGDState``
             * *function that performs one SVGD step.*


.. py:function:: rbf_kernel(x, y, length_scale=1)

   Radial basis function (RBF / squared-exponential) kernel.

   :param x: First particle (PyTree).
   :param y: Second particle (PyTree).
   :param length_scale: Bandwidth of the kernel. Larger values produce smoother kernels.

   :rtype: Scalar kernel evaluation ``exp(-||x - y||^2 / length_scale)``.


.. py:function:: update_median_heuristic(state: SVGDState) -> SVGDState

   Median heuristic for setting the bandwidth of RBF kernels.

   A reasonable middle-ground for choosing the ``length_scale`` of the RBF
   kernel is to pick the empirical median of the squared distance between
   particles. This strategy is called the median heuristic.

   :param state: Current SVGDState whose particles are used to compute the heuristic.

   :returns: * Updated SVGDState with ``kernel_parameters["length_scale"]`` set via
             * *the median heuristic.*


.. py:function:: as_top_level_api(grad_logdensity_fn: collections.abc.Callable, optimizer, kernel: collections.abc.Callable = rbf_kernel, update_kernel_parameters: collections.abc.Callable = update_median_heuristic)

   Implements the (basic) user interface for the svgd algorithm :cite:p:`liu2016stein`.

   :param grad_logdensity_fn: gradient, or an estimate, of the target log density function to samples approximately from
   :param optimizer: Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
   :param kernel: positive semi definite kernel
   :param update_kernel_parameters: function that updates the kernel parameters given the current state of the particles

   :rtype: A ``SamplingAlgorithm``.


