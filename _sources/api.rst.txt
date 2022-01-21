Common Kernels
==============

.. currentmodule:: blackjax

.. autosummary::

  hmc
  nuts
  rmh
  tempered_smc

HMC
~~~

.. automodule:: blackjax.hmc
   :members: HMCInfo, kernel, new_state

NUTS
~~~~

.. automodule:: blackjax.nuts
   :members: NUTSInfo, kernel, new_state

RMH
~~~

.. automodule:: blackjax.rmh
    :members:
    :undoc-members:

Tempered SMC
~~~~~~~~~~~~

.. automodule:: blackjax.tempered_smc
   :members: TemperedSMCState, adaptive_tempered_smc, tempered_smc


Adaptation
==========


Stan full warmup
~~~~~~~~~~~~~~~~

.. currentmodule:: blackjax

.. automodule:: blackjax.stan_warmup
   :members: run

Step-size adataptation
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: blackjax.adaptation.step_size

.. autofunction:: dual_averaging_adaptation

.. autofunction:: find_reasonable_step_size

Mass matrix adataptation
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: blackjax.adaptation.mass_matrix

.. autofunction:: mass_matrix_adaptation

Diagnostics
===========

.. currentmodule:: blackjax.diagnostics

.. autosummary::

  effective_sample_size
  potential_scale_reduction

Effective sample size
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: effective_sample_size


Potential scale reduction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: potential_scale_reduction
