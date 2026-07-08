blackjax.progress_reader
========================

.. py:module:: blackjax.progress_reader

.. autoapi-nested-parse::

   Standalone reader for file-based progress output.

   Usage: python -m blackjax.progress_reader /tmp/bjx_progress.txt



Functions
---------

.. autoapisummary::

   blackjax.progress_reader.read_progress
   blackjax.progress_reader.main


Module Contents
---------------

.. py:function:: read_progress(path)

   Read '<step> <total>' from ``path``.

   :returns: * ``(step, total)`` tuple of ints, or ``None`` if the file does not exist
             * or does not yet contain a parseable ``"<step> <total>"`` payload (e.g. it
             * *was read mid-write).*


.. py:function:: main(argv=None)

