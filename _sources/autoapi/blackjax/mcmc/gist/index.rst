blackjax.mcmc.gist
==================

.. py:module:: blackjax.mcmc.gist

.. autoapi-nested-parse::

   Back-compatibility shim — use :mod:`blackjax.mcmc.composed` instead.

   .. deprecated::
       Importing from ``blackjax.mcmc.gist`` is deprecated. The module has been
       reorganised into the :mod:`blackjax.mcmc.composed` sub-package (the
       general GIST kernel spine now lives in
       :mod:`blackjax.mcmc.composed._seam`). All public names are still
       importable; update your imports to use the new location:

       - ``from blackjax.mcmc.composed import GISTState``
       - ``from blackjax.mcmc.composed import GISTInfo``
       - ``from blackjax.mcmc.composed import init``
       - ``from blackjax.mcmc.composed import build_kernel``
       - ``from blackjax.mcmc.composed._seam import as_top_level_api``



