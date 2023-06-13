:py:mod:`blackjax.vi.svgd`
==========================

.. py:module:: blackjax.vi.svgd


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.vi.svgd.svgd



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.vi.svgd.rbf_kernel
   blackjax.vi.svgd.update_median_heuristic



.. py:function:: rbf_kernel(x, y, length_scale=1)


.. py:function:: update_median_heuristic(state: SVGDState) -> SVGDState

   Median heuristic for setting the bandwidth of RBF kernels.

   A reasonable middle-ground for choosing the `length_scale` of the RBF kernel
   is to pick the empirical median of the squared distance between particles.
   This strategy is called the median heuristic.


.. py:class:: svgd


   Implements the (basic) user interface for the svgd algorithm.

   :param grad_logdensity_fn: gradient, or an estimate, of the target log density function to samples approximately from
   :param optimizer: Optax compatible optimizer, which conforms to the `optax.GradientTransformation` protocol
   :param kernel: positive semi definite kernel
   :param update_kernel_parameters: function that updates the kernel parameters given the current state of the particles

   :rtype: A ``SamplingAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: build_kernel

      


