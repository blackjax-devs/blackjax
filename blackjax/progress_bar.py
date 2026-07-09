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
"""A ``jaxtap``-powered progress bar for BlackJAX.

The :func:`progress_bar` context manager subscribes to scan-step events
emitted by ``jaxtap.record()`` (the A-shell monkeypatch form) and feeds
them into a :class:`ProgressState` display loop.  The public API, display
behaviour, and :mod:`blackjax.progress_reader` file format are unchanged
from the #964 implementation; only the *emission* side changes:

* **Deleted** (scan-patch machinery):  ``_patched_scan``, ``_inject_progress``,
  ``_underlying_scan``/``_session_scan``, ``_registry_lock``,
  ``_progress_registry``, depth counter.  These are now jaxtap's
  responsibility.
* **Kept** (display side): :class:`ProgressState` (all fields), the
  ``_render_loop`` with multi-phase reset, ``_step_callback``'s totality
  invariant, zero-step ``UserWarning``, ``output_file`` atomic write +
  reader format.

Import cost: ``import blackjax`` does **not** import ``jaxtap``.  The
import happens lazily inside :func:`progress_bar`'s ``__enter__`` so a
plain ``pip install blackjax`` install continues to work.
"""
import os
import threading
import time
import warnings
from contextlib import contextmanager

__all__ = ["progress_bar"]


class ProgressState:
    """Mutable, shared state for one active :func:`progress_bar` context.

    A plain Python object updated by the jaxtap ``on_step`` callback — it
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
        jaxtap ``on_step`` callback path that ultimately goes through a
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
                # first. `n_steps` is written before that phase's first
                # callback fires (see _on_step in progress_bar()), so the
                # only possible race is a single stale tick where this loop
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
    algorithm being run.  Emission is powered by ``jaxtap`` (the
    ``blackjax[progress]`` optional extra).

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
    Works under ``jax.vmap`` -- only the outermost ``jax.lax.scan`` is
    instrumented (nested scans inside the body are not tapped), so the
    progress bar fires exactly once per real outer step regardless of
    chain count.  The mechanism: jaxtap inserts a step counter as
    ``jnp.int32(0)`` in the scan carry -- a compile-time constant that
    depends only on itself (``step + 1``), so it stays unbatched under
    vmap even when the rest of the carry is batched; ``jax.debug.callback``
    with an unbatched argument fires once per step for the entire batch.

    JIT-cache staleness has two distinct cases:

    * **Function compiled inside an earlier context, called inside a new
      one** (cache-hit in ctx2): jaxtap bakes ``_dynamic_router``, a
      module-level singleton, into XLA artifacts rather than a closure.
      A cache-hit in ctx2 routes events to ctx2's live recorder at call
      time -- the bar works, no zero-step warning fires.  Keeping one
      context open across repeated calls to the same compiled function is
      still the safest pattern (avoids any edge cases from mismatched
      ``select`` config baked at trace time).
    * **Function compiled before entering any context**: no callback is
      baked in at all (``jax.lax.scan`` was the original at compile time).
      A later context sees zero events and the zero-step warning fires.
      Fix: call ``jax.clear_caches()`` before entering the context to
      force a retrace.

    Under ``jax.checkpoint`` combined with differentiation, the wrapped scan
    body -- including the injected callback -- runs twice per logical step
    (primal pass + backward-pass recompute), so the displayed step
    count/rate appear roughly doubled; computed values and gradients are
    unaffected.

    **Process-global patch -- scope and boundaries.** This context manager
    monkeypatches ``jax.lax.scan`` (via ``jaxtap``) for its duration:

    * Only use the ``with`` form. A manually ``__enter__``ed context (e.g.
      in an interactive session) that is never ``__exit__``ed leaves
      ``jax.lax.scan`` patched for the rest of the process/kernel session.
      Call ``jaxtap.emergency_restore()`` to recover.
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
    * A user's own ``jaxtap.record()`` context opened *inside* this context
      wins attribution for scans run within its block (innermost-context
      wins semantics from jaxtap); the progress bar will not update for
      those scans.

    Examples
    --------
    .. code::

        with blackjax.progress_bar(label="NUTS warmup"):
            (state, params), _ = blackjax.window_adaptation(
                blackjax.nuts, logdensity_fn
            ).run(rng_key, initial_position, 1000)
    """
    try:
        import jaxtap as tap
    except ImportError as exc:
        raise ImportError(
            "blackjax.progress_bar requires the 'progress' optional extra. "
            "Install it with:  pip install 'blackjax[progress]'"
        ) from exc

    state = ProgressState(label=label, print_rate=print_rate, output_file=output_file)

    def _on_step(event):
        """Route a jaxtap TapEvent to the ProgressState display.

        Called on the host for every scan step of the outermost intercepted
        scan.  ``event.total`` is the scan length (known at trace time and
        delivered by jaxtap); ``event.step`` is the 0-based iteration index
        (already a Python int, courtesy of jaxtap's ``step_.item()`` call in
        its host closure).

        For ``reverse=True`` scans, jaxtap's step counter still runs
        0, 1, ..., N-1 (it lives in the carry, not in xs), so the display
        naturally shows an ascending sequence -- no flip needed at this layer.

        For sequential multi-phase runs (e.g. ``mclmc_find_L_and_step_size``'s
        ~5 warmup scans), each phase delivers ``event.total`` for its own
        length.  Updating ``n_steps`` before calling ``_step_callback`` ensures
        the render loop sees the new total before the phase-reset signal
        (``step == 0``) arrives.
        """
        if event.total is not None:
            state.n_steps = event.total
        state._step_callback(event.step)

    # tap.record() A-shell:
    #   select=lambda _: ()  → progress idiom: only the step counter (a Python
    #                          int after jaxtap's .item() call) crosses the host
    #                          boundary; zero bytes of carry data shipped.
    #   ops=("scan",)        → instrument jax.lax.scan only, not while_loop
    #                          (matches current behaviour).
    #   max_depth=0          → emit carry taps only for the outermost intercepted
    #                          scan (depth 0 = path has 0 '/' separators); nested
    #                          scans inside the body are walked by the B-core but
    #                          do not emit events, so they never fire _on_step.
    tap_ctx = tap.record(
        on_step=_on_step,
        select=lambda _: (),
        ops=("scan",),
        max_depth=0,
    )

    state._start_display()
    tap_ctx.__enter__()
    try:
        yield state
    finally:
        state.closed = True
        state._stop_event.set()
        if state._display_thread is not None:
            state._display_thread.join(timeout=2.0)
        # Clear before removing: a jit-cached function traced inside this
        # context keeps its callback baked in after exit (see the docstring
        # Notes), and a post-exit call would otherwise resurrect the file
        # we are about to delete via that stale on_step callback.
        state.output_file = None
        tap_ctx.__exit__(None, None, None)
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
