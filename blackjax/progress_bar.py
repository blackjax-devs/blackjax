# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A ``jax.lax.scan``-monkeypatching progress bar for BlackJAX.

Supersedes the old ``gen_scan_fn``/``progress_bar_scan`` mechanism (which
augmented the scan *carry* with a ``chain_id`` and broke under ``jax.vmap``
with a shape mismatch in its ``io_callback`` result -- see issue #927). This
module instead augments the scan ``xs`` with a step index and fires a
``jax.debug.callback`` (which returns ``None``, so there is no result shape to
mismatch under ``vmap``); the carry is never touched, so it composes cleanly
with any transformation that batches or otherwise rewrites the carry.
"""
import os
import threading
import time
import uuid
from contextlib import contextmanager

import jax
import jax.numpy as jnp

__all__ = ["progress_bar"]

_original_scan = jax.lax.scan
_scan_depth = threading.local()
_progress_registry: dict = {}  # uuid -> ProgressState


def _patched_scan(f, init, xs=None, length=None, **kwargs):
    """Drop-in replacement for ``jax.lax.scan`` installed while a
    :func:`progress_bar` context is active.

    Only the *outermost* ``lax.scan`` encountered at trace time is
    instrumented -- nested scans (e.g. a thinning inner loop) fall through to
    the original ``lax.scan`` unmodified. Outermost-ness is tracked with a
    thread-local depth counter that increments for the duration of tracing
    ``f`` (which happens synchronously inside the call to the original
    ``lax.scan`` below), so any ``lax.scan`` call made from within ``f`` sees
    a depth > 0 and is left alone.
    """
    depth = getattr(_scan_depth, "value", 0)
    _scan_depth.value = depth + 1
    try:
        active = list(_progress_registry.values())
        if depth == 0 and active:
            # Innermost-active-context wins when progress_bar() contexts are
            # nested (rare, but well-defined: latest entered = latest in the
            # registry's insertion order).
            state = active[-1]
            return _inject_progress(f, init, xs, length, state, **kwargs)
        else:
            return _original_scan(f, init, xs=xs, length=length, **kwargs)
    finally:
        _scan_depth.value = depth


def _inject_progress(f, init, xs, length, state, **kwargs):
    """Wrap ``f`` so a step index is threaded through ``xs`` (never the
    carry) and fire a callback on every step."""
    if length is not None:
        n = length
    elif xs is not None:
        leaves = jax.tree.leaves(xs)
        if not leaves:
            return _original_scan(f, init, xs=xs, length=length, **kwargs)
        n = leaves[0].shape[0]
    else:
        return _original_scan(f, init, xs=xs, length=length, **kwargs)

    state.n_steps = n
    indices = jnp.arange(n)

    def f_wrapped(carry, augmented_x):
        original_x, idx = augmented_x
        # debug.callback returns None -> no result_shape_dtypes -> no shape
        # mismatch under vmap (unlike io_callback with a scalar result).
        jax.debug.callback(state._step_callback, idx, ordered=False)
        return f(carry, original_x)

    return _original_scan(f_wrapped, init, xs=(xs, indices), length=None, **kwargs)


class ProgressState:
    """Mutable, shared state for one active :func:`progress_bar` context.

    A plain Python object closed over by the ``jax.debug.callback`` -- it
    never enters the traced computation itself, only the step index does.
    """

    def __init__(self, label, print_rate, output_file):
        self.label = label
        self.n_steps = 0
        self.print_rate = print_rate  # None -> resolved lazily
        self.output_file = output_file
        self.current_step = 0
        self._bar = None
        self._tqdm_cls = None
        self._stop_event = threading.Event()
        self._display_thread = None

    def _resolved_print_rate(self):
        if self.print_rate is not None:
            return self.print_rate
        return max(1, self.n_steps // 20)

    def _step_callback(self, idx):
        """Runs on the host once per scan step (outermost scan only)."""
        step = int(idx)
        rate = self._resolved_print_rate()
        if step % rate == 0 or step == self.n_steps - 1:
            # Track the max seen so the bar stays monotone even if steps
            # arrive out of order (e.g. under some future scheduling).
            if step > self.current_step or step == 0:
                self.current_step = step
            if self.output_file:
                tmp = self.output_file + ".tmp"
                with open(tmp, "w") as fh:
                    fh.write(f"{step} {self.n_steps}")
                os.replace(tmp, self.output_file)  # atomic

    def _start_display(self):
        from tqdm.auto import tqdm

        self._tqdm_cls = tqdm
        self._display_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._display_thread.start()

    def _render_loop(self):
        bar = None
        last = -1
        while not self._stop_event.is_set():
            # n_steps is unknown until the outermost scan is traced, so wait
            # for it before creating the bar.
            if self.n_steps > 0 and bar is None:
                bar = self._tqdm_cls(total=self.n_steps, desc=self.label, unit="step")
            if bar is not None:
                cur = self.current_step
                if cur != last:
                    bar.n = cur + 1
                    bar.refresh()
                    last = cur
            time.sleep(0.1)
        if bar is not None:
            bar.n = self.n_steps
            bar.refresh()
            bar.close()


@contextmanager
def progress_bar(label="BlackJAX", print_rate=None, output_file=None):
    """Add a progress bar to any BlackJAX sampling call.

    Automatically detects the outermost ``jax.lax.scan`` in the wrapped code
    and injects a step counter, without requiring any parameter on the
    algorithm being run.

    Parameters
    ----------
    label
        Display label for the progress bar.
    print_rate
        Update every ``print_rate`` steps. Defaults to ``max(1, num_steps // 20)``.
    output_file
        If given, atomically writes ``"<step> <total>"`` to this path on each
        update. Display it from another terminal with
        ``python -m blackjax.progress_reader <path>``.

    Notes
    -----
    Works under ``jax.vmap`` -- the bar shows the maximum step seen across
    chains instead of crashing (fixes #927).

    Only the outermost ``jax.lax.scan`` traced *while this context is
    active* is instrumented. Because ``jax.jit`` caches compiled functions, a
    function traced once outside this context (and cached) will not show a
    bar even if later called from inside it -- and conversely a function
    traced inside this context keeps its callback baked in even after the
    context exits. Force a retrace (e.g. by clearing the JIT cache, or by not
    pre-jitting) if you need the bar to reliably appear.

    Examples
    --------
    .. code::

        with blackjax.progress_bar(label="NUTS warmup"):
            (state, params), _ = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn
            ).run(rng_key, initial_position, 1000)
    """
    key = str(uuid.uuid4())
    state = ProgressState(label=label, print_rate=print_rate, output_file=output_file)
    _progress_registry[key] = state
    jax.lax.scan = _patched_scan
    state._start_display()
    try:
        yield state
    finally:
        state._stop_event.set()
        if state._display_thread is not None:
            state._display_thread.join(timeout=2.0)
        del _progress_registry[key]
        if not _progress_registry:
            jax.lax.scan = _original_scan
        # Clear before removing: a jit-cached function traced inside this
        # context keeps its callback baked in after exit (see the docstring
        # Notes), and a post-exit call would otherwise resurrect the file
        # we are about to delete via that stale _step_callback.
        state.output_file = None
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
