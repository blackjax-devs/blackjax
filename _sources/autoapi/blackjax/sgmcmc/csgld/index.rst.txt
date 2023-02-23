:py:mod:`blackjax.sgmcmc.csgld`
===============================

.. py:module:: blackjax.sgmcmc.csgld

.. autoapi-nested-parse::

   Public API for the Contour Stochastic gradient Langevin Dynamics kernel :cite:p:`deng2020contour,deng2022interacting`.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.csgld.ContourSGLDState



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.sgmcmc.csgld.init
   blackjax.sgmcmc.csgld.kernel



.. py:class:: ContourSGLDState



   State of the Contour SgLD algorithm.

   :param position: Current position in the sample space.
   :param energy_pdf: Vector with `m` non-negative values that sum to 1. The `i`-th value
                      of the vector is equal to :math:`\int_{S_1} \pi(\mathrm{d}x)` where
                      :math:`S_i` is the `i`-th energy partition.
   :param energy_idx: Index `i` such that the current position belongs to :math:`S_i`.

   .. py:attribute:: position
      :type: blackjax.types.PyTree

      

   .. py:attribute:: energy_pdf
      :type: blackjax.types.Array

      

   .. py:attribute:: energy_idx
      :type: int

      


.. py:function:: init(position: blackjax.types.PyTree, num_partitions=512)


.. py:function:: kernel(num_partitions=512, energy_gap=10, min_energy=0) -> Callable

   :param num_partitions: The number of partitions we divide the energy landscape into.
   :param energy_gap: The difference in energy :math:`\Delta u` between the successive
                      partitions. Can be determined by running e.g. an optimizer to determine
                      the range of energies. `num_partition` * `energy_gap` should match this
                      range.
   :param min_energy: A rough estimate of the minimum energy in a dataset, which should be
                      strictly smaller than the exact minimum energy! e.g. if the minimum
                      energy of a dataset is 3456, we can set min_energy to be any value
                      smaller than 3456. Set it to 0 is acceptable, but not efficient enough.
                      the closer the gap between min_energy and 3456 is, the better.


