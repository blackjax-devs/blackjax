blackjax.adaptation.meta_adaptation
===================================

.. py:module:: blackjax.adaptation.meta_adaptation

.. autoapi-nested-parse::

   Back-compatibility shim — use :mod:`blackjax.adaptation.meta` instead.

   .. deprecated::
       Importing from ``blackjax.adaptation.meta_adaptation`` is deprecated.
       The module has been reorganised into the
       :mod:`blackjax.adaptation.meta` sub-package.  All public names are
       still importable; update your imports to use the new location:

       - ``from blackjax.adaptation.meta import build_meta_adaptation_core``
       - ``from blackjax.adaptation.meta import build_multi_chain_meta_core``
       - ``from blackjax.adaptation.meta import extract_meta_verdict``
       - ``from blackjax.adaptation.meta import extract_multi_chain_verdict``
       - ``from blackjax.adaptation.meta._calibration import _ASSUMED_AVG_LEAPFROGS_PER_STEP``
       - … (see :mod:`blackjax.adaptation.meta` for the full public surface)



