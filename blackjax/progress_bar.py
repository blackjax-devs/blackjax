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
import warnings
from contextlib import contextmanager

import jax
import jax.numpy as jnp

__all__ = ["progress_bar"]

_original_scan = jax.lax.scan
_scan_depth = threading.local()
_progress_registry: dict = {}  # uuid -> ProgressState

# Guards the registry empty<->non-empty transitions and the _session_scan
# capture/restore below (progress_bar() enter/exit only -- NOT held across
# the `yield`, and NOT touched by _patched_scan/_inject_progress on the
# scan-tracing hot path). Without this, two threads racing to be the FIRST
# entrant could both observe an empty registry before either inserts: the
# second to actually patch `jax.lax.scan` would have the FIRST's bare
# `_patched_scan` visible as "the current jax.lax.scan" and capture that as
# `_session_scan`, making `_underlying_scan()` return `_patched_scan` itself
# -- infinite self-recursion on every subsequent pass-through call. Holding
# this lock around the narrow enter/exit critical section (a dict insert/del
# plus one attribute read-or-write) serializes exactly that race at
# essentially zero cost, since entering/exiting a context is rare compared
# to the scans that run inside it.
_registry_lock = threading.Lock()

# The scan implementation captured on the empty->non-empty registry
# transition (i.e. whatever `jax.lax.scan` pointed to when the FIRST
# concurrently-active progress_bar() context was entered). This may be a
# third-party monkeypatch rather than `_original_scan` -- capturing and
# restoring THIS value (instead of unconditionally restoring the import-time
# original) avoids clobbering a foreign patch that predates or outlives our
# own. `None` means no session is currently open.
_session_scan = None


def _underlying_scan():
    """The scan implementation to delegate to for anything we do not
    instrument (nested scans, no active context, etc).

    Falls back to a captured foreign patch if one was installed before our
    session started, so a pre-existing third-party `jax.lax.scan` patch
    keeps functioning for calls made *inside* our context instead of being
    silently bypassed -- true chaining, not just non-clobbering on exit.
    """
    return _session_scan if _session_scan is not None else _original_scan


def _patched_scan(f, init, xs=None, length=None, **kwargs):
    """Drop-in replacement for ``jax.lax.scan`` installed while a
    :func:`progress_bar` context is active.

    Only the *outermost* ``lax.scan`` encountered at trace time is
    instrumented -- nested scans (e.g. a thinning inner loop) fall through to
    the underlying ``lax.scan`` unmodified. Outermost-ness is tracked with a
    thread-local depth counter that increments for the duration of tracing
    ``f`` (which happens synchronously inside the call to the underlying
    ``lax.scan`` below), so any ``lax.scan`` call made from within ``f`` sees
    a depth > 0 and is left alone.

    Attribution when multiple ``progress_bar()`` contexts are simultaneously
    active (e.g. one per thread): with exactly one context active, ANY
    calling thread is attributed to it (this is what makes "enter the
    context on the main thread, run the scan on a worker thread" work). Once
    a second context is active, attribution requires an exact match between
    the calling thread and the context's *owner* thread (the thread that
    called ``__enter__``); a scan from any other thread -- including a
    genuine bystander or another context's own delegate -- falls through
    with no bar rather than risk guessing wrong. See the module docstring
    Notes for the fully disclosed decision table.
    """
    depth = getattr(_scan_depth, "value", 0)
    _scan_depth.value = depth + 1
    try:
        active = list(_progress_registry.values())
        if depth == 0 and active:
            if len(active) >= 2:
                here = threading.get_ident()
                owned = [s for s in active if s.owner_thread == here]
                if not owned:
                    return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
                state = owned[-1]
            else:
                state = active[-1]
            return _inject_progress(f, init, xs, length, state, **kwargs)
        else:
            return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
    finally:
        _scan_depth.value = depth


def _inject_progress(f, init, xs, length, state, **kwargs):
    """Wrap ``f`` so a step index is threaded through ``xs`` (never the
    carry) and fire a callback on every step."""
    if length is not None:
        n = length
        if xs is not None:
            leaves = jax.tree.leaves(xs)
            if leaves and leaves[0].shape[0] != n:
                # length disagrees with xs's own leading dimension -- let
                # the underlying scan raise its own actionable error
                # ("scan got `length` argument of ... which disagrees
                # with ...") instead of us augmenting xs and having JAX
                # blame an error on our internally-built pytree.
                return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
    elif xs is not None:
        leaves = jax.tree.leaves(xs)
        if not leaves:
            return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
        n = leaves[0].shape[0]
    else:
        return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)

    state.n_steps = n
    reverse = kwargs.get("reverse", False)
    indices = jnp.arange(n)

    def f_wrapped(carry, augmented_x):
        original_x, idx = augmented_x
        # `display_idx` is a pure function of the static `n` and the
        # unbatched `idx` (jnp.arange(n)) -- it never touches the batched
        # axis under vmap, which is what makes the callback fire exactly
        # once per real step regardless of chain count (see
        # _step_callback's docstring and the module Notes). If a future
        # change ever passes a value here that DOES depend on the vmapped
        # input, JAX's vmap batching rule will unroll it into one call per
        # batch element per step, silently breaking that guarantee --
        # re-validate vmap behavior before adding any batched argument.
        display_idx = (n - 1 - idx) if reverse else idx
        jax.debug.callback(state._step_callback, display_idx, ordered=False)
        return f(carry, original_x)

    return _underlying_scan()(f_wrapped, init, xs=(xs, indices), length=None, **kwargs)


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
        self.owner_thread = threading.get_ident()
        self.closed = False
        self._output_file_warned = False
        self._callback_warned = False
        self._bar = None
        self._tqdm_cls = None
        self._stop_event = threading.Event()
        self._display_thread = None

    def _resolved_print_rate(self):
        if self.print_rate is not None:
            return self.print_rate
        return max(1, self.n_steps // 20)

    def _step_callback(self, idx):
        """Runs on the host once per scan step (outermost scan only).

        INVARIANT: this function must never raise. It executes inside a
        ``jax.debug.callback``; an exception here crosses back into the XLA
        runtime as a fatal ``JaxRuntimeError`` and kills the ENTIRE traced
        computation, not just the progress display -- e.g. an
        ``output_file`` write failure (a permission change, a full disk, an
        NFS hiccup) must never be allowed to crash an in-progress sampling
        run. The outer ``except Exception`` is intentionally broad: the
        invariant is unconditional and the specific ways a host callback can
        fail cannot be enumerated in advance. The inner ``except OSError``
        handles the one recurring, retryable-looking failure (the
        ``output_file`` write) so we stop paying a doomed write on every
        remaining step instead of relying on the outer catch every time.

        Both handlers mutate state (the warned-once flag, disabling
        ``output_file``) BEFORE attempting to warn, and wrap the
        ``warnings.warn`` call itself in its own ``try/except``: if the
        active warnings filter promotes ``UserWarning`` to an error (this
        project's own ``pytest.ini`` does), the courtesy warning must not be
        allowed to become the very exception this function exists to
        prevent -- the never-raise invariant outranks the warning.
        """
        if self.closed:
            return
        try:
            step = int(idx)
            rate = self._resolved_print_rate()
            if rate <= 0:  # print_rate=0 is an innocent typo, not a crash
                rate = 1
            if step % rate == 0 or step == self.n_steps - 1:
                # Track the max seen so the bar stays monotone even if steps
                # arrive out of order (e.g. under some future scheduling).
                # `step == 0` is an unconditional reset: on a multi-phase
                # run (e.g. mclmc_find_L_and_step_size's ~5 sequential
                # warmup scans), this is what tells _render_loop a new
                # phase has started even when the new phase happens to have
                # the same length as the previous one (so `n_steps` alone
                # would not change) -- see _render_loop's `cur < last`
                # check.
                if step > self.current_step or step == 0:
                    self.current_step = step
                if self.output_file:
                    try:
                        tmp = self.output_file + ".tmp"
                        with open(tmp, "w") as fh:
                            fh.write(f"{step} {self.n_steps}")
                        os.replace(tmp, self.output_file)  # atomic
                    except OSError as e:
                        already_warned = self._output_file_warned
                        self._output_file_warned = True
                        self.output_file = None
                        if not already_warned:
                            try:
                                warnings.warn(
                                    "blackjax.progress_bar: disabling "
                                    f"output_file after a write failure "
                                    f"({e!r}) -- the progress bar itself is "
                                    "unaffected.",
                                    stacklevel=2,
                                )
                            except Exception:
                                # warnings can be promoted to errors by the
                                # active filter; the never-raise invariant
                                # outranks the courtesy.
                                pass
        except Exception as e:  # noqa: BLE001 -- see invariant above
            already_warned = self._callback_warned
            self._callback_warned = True
            if not already_warned:
                try:
                    warnings.warn(
                        "blackjax.progress_bar: internal callback error "
                        f"({e!r}) -- further progress updates for this "
                        "context are disabled, but the underlying "
                        "computation is unaffected.",
                        stacklevel=2,
                    )
                except Exception:
                    # warnings can be promoted to errors by the active
                    # filter; the never-raise invariant outranks the
                    # courtesy.
                    pass

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
            if self.n_steps > 0:
                cur = self.current_step
                # Detect a new phase on a multi-phase run (e.g.
                # mclmc_find_L_and_step_size's ~5 sequential warmup scans of
                # different lengths, immediately followed by a sampling
                # scan): `n_steps` changing is the common case, but two
                # consecutive phases can share the same length, in which
                # case `n_steps` alone would not signal the restart -- `cur
                # < last` catches that (a new phase's step 0 always resets
                # `current_step`, see _step_callback). Either signal resets
                # the bar so its total/elapsed-timer/rate stats reflect the
                # new phase instead of freezing at the first phase's total
                # (the observed bug) or overflowing into tqdm's unbounded
                # counter mode once a later phase runs longer than the
                # first. `n_steps` is written at trace time, strictly
                # before that phase's first callback can fire, so the only
                # possible race is a single stale tick where this loop
                # observes the new `n_steps` before `current_step` has been
                # reset to 0 by that phase's first callback -- harmless,
                # self-corrects on the very next 0.1s tick.
                if bar is None:
                    bar = self._tqdm_cls(
                        total=self.n_steps, desc=self.label, unit="step"
                    )
                    last = -1
                elif self.n_steps != bar.total or cur < last:
                    bar.reset(total=self.n_steps)
                    last = -1
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
    Works under ``jax.vmap`` -- the injected step index is a compile-time
    constant (``jnp.arange(n)``) that never depends on the batched axis, so
    it stays unbatched and the callback fires exactly once per real scan
    step regardless of chain count, instead of the shape-mismatch crash the
    old ``io_callback``-based mechanism had (fixes #927).

    Only the outermost ``jax.lax.scan`` traced *while this context is
    active* is instrumented. The retrace boundary that actually matters is
    repeated calls to an already-traced/jit'd/checkpointed function across
    DIFFERENT ``progress_bar()`` contexts -- a fresh ``jax.jit(fn)`` wrapper
    is NOT itself a retrace boundary (JAX's jit cache is keyed by function
    identity, not by whether a progress_bar context happens to be active),
    so calling the SAME compiled function again inside a *later* context
    reuses the cached trace and its callback stays baked in from whichever
    context (if any) was active the FIRST time it was traced. Safe pattern:
    keep ONE ``progress_bar()`` context open across all repeated calls to
    the same compiled function. Workaround if you must re-enter: force a
    retrace with ``jax.clear_caches()`` before the next call. A stale
    baked-in callback also has a small phantom cost (~32us/step of host
    dispatch) on every future execution into an already-closed, inert
    state.

    Under ``jax.checkpoint`` combined with differentiation, the wrapped scan
    body -- including the injected callback -- runs twice per logical step
    (primal pass + backward-pass recompute), so the displayed step
    count/rate appear roughly doubled; computed values and gradients are
    unaffected.

    **Process-global patch -- scope and boundaries.** This context manager
    monkeypatches ``jax.lax.scan`` for its duration, which has consequences
    beyond the ``with`` block itself:

    * Only use the ``with`` form. A manually ``__enter__``ed context (e.g.
      in an interactive session) that is never ``__exit__``ed leaves
      ``jax.lax.scan`` patched for the rest of the process/kernel session.
    * With exactly one context active, ANY calling thread's top-level scan
      is attributed to it -- this is what makes "enter the context on the
      main thread, run sampling on a worker thread/executor" work, but it
      also means a genuinely unrelated bystander scan on another thread is
      indistinguishable from that pattern and gets the same bar. With two
      or more contexts simultaneously active, attribution instead requires
      an exact owner-thread match (the thread that called ``__enter__``); a
      scan from any other thread -- including a legitimate delegate of one
      of the active contexts -- falls through with no bar rather than risk
      misattributing it to the wrong context.
    * Under ``jax.shard_map``/multi-device execution the callback fires
      once per device per step (``N`` times the host-callback overhead);
      the bar can reach 100% while a slower shard is still running.
    * A ``functools.partial(jax.lax.scan, ...)`` captured before the
      context is entered keeps a reference to the un-patched function and
      silently bypasses the bar entirely (no error, no bar).
    * If two contexts share the same ``output_file`` path, their writes
      interleave and corrupt the file -- use a unique path per context.

    Examples
    --------
    .. code::

        with blackjax.progress_bar(label="NUTS warmup"):
            (state, params), _ = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn
            ).run(rng_key, initial_position, 1000)
    """
    global _session_scan

    key = str(uuid.uuid4())
    state = ProgressState(label=label, print_rate=print_rate, output_file=output_file)
    with _registry_lock:
        if not _progress_registry:  # empty -> non-empty: this is the session opener
            candidate = jax.lax.scan  # whatever is installed now (foreign or original)
            # Belt-and-braces: the lock above already prevents the race
            # that could make this true (two threads both observing an
            # empty registry before either inserts), but guard the
            # invariant directly anyway -- capturing our OWN bare patch
            # here would make _underlying_scan() return _patched_scan
            # itself, i.e. infinite self-recursion on every pass-through
            # call. A foreign wrapper installed AROUND _patched_scan is a
            # different object and would still chain through correctly.
            if candidate is not _patched_scan:
                _session_scan = candidate
        _progress_registry[key] = state
        jax.lax.scan = _patched_scan
    state._start_display()
    try:
        yield state
    finally:
        state.closed = True
        state._stop_event.set()
        if state._display_thread is not None:
            state._display_thread.join(timeout=2.0)
        with _registry_lock:
            del _progress_registry[key]
            if not _progress_registry:  # non-empty -> empty: session closer
                if jax.lax.scan is _patched_scan:
                    jax.lax.scan = _session_scan
                # Else: something else replaced jax.lax.scan while we were
                # active -- leave it alone rather than clobbering a foreign
                # patch installed during our session.
                _session_scan = None
        # Clear before removing: a jit-cached function traced inside this
        # context keeps its callback baked in after exit (see the docstring
        # Notes), and a post-exit call would otherwise resurrect the file
        # we are about to delete via that stale _step_callback.
        state.output_file = None
        if output_file and os.path.exists(output_file):
            os.remove(output_file)
        if state.n_steps == 0:
            warnings.warn(
                "blackjax.progress_bar: no scan step was ever observed "
                "inside this context. Either the wrapped code never called "
                "jax.lax.scan, or the function was already "
                "jit-compiled/traced before this context was entered "
                "(jax.jit's cache is keyed by function identity, so a "
                "cache hit skips retracing and the injected callback never "
                "gets baked in). Keep ONE progress_bar() context open "
                "across repeated calls to the same compiled function, or "
                "force a retrace with jax.clear_caches().",
                stacklevel=2,
            )
