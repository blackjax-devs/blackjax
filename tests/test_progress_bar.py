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
"""Unit tests for the jaxtap-powered progress bar."""
import os
import stat
import tempfile
import threading
import warnings

import chex
import jax
import jax.numpy as jnp

# M1 adaptation: _original_scan / _patched_scan / _session_scan were deleted
# from blackjax.progress_bar when the homegrown scan-patch machinery was
# replaced by jaxtap.  These are now jaxtap's internal globals.  Tests that
# need to assert restoration / self-capture / non-LIFO exit semantics import
# them from jaxtap._ashell, where the identical invariants are maintained.
# Justification: patch lifecycle (install / chain / restore) is now jaxtap's
# responsibility; the user-visible semantics (scan restored after exit, foreign
# patch chained and restored, >=2-context owner-affinity) are identical.
import jaxtap._ashell as _jaxtap_ashell
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.progress_bar import ProgressState
from blackjax.progress_reader import read_progress
from blackjax.util import run_inference_algorithm

_original_scan = _jaxtap_ashell._original_scan
_patched_scan = _jaxtap_ashell._patched_scan

from tests.fixtures import BlackJAXTest, std_normal_logdensity


class ProgressBarTest(BlackJAXTest):
    def test_basic(self):
        """``run_inference_algorithm`` (NUTS, 100 steps, 2-dim gaussian)
        completes inside the context manager and the callback fires."""
        algorithm = blackjax.nuts(
            std_normal_logdensity, step_size=0.1, inverse_mass_matrix=jnp.eye(2)
        )
        with blackjax.progress_bar(label="test") as state:
            final_state, _ = run_inference_algorithm(
                rng_key=self.next_key(),
                inference_algorithm=algorithm,
                num_steps=100,
                initial_position=jnp.zeros(2),
            )
            jax.block_until_ready(final_state)

        self.assertEqual(state.n_steps, 100)
        self.assertGreater(state.current_step, 0)

    def test_vmap_no_crash(self):
        """Regression test for #927: ``jax.vmap`` over 2 chains of a
        scan-based sampler must not crash inside the context manager."""
        algorithm = blackjax.hmc(
            std_normal_logdensity,
            step_size=0.1,
            inverse_mass_matrix=jnp.eye(2),
            num_integration_steps=5,
        )

        def run(key, position):
            return run_inference_algorithm(
                rng_key=key,
                inference_algorithm=algorithm,
                num_steps=50,
                initial_position=position,
            )

        keys = jax.random.split(self.next_key(), 2)
        positions = jnp.zeros((2, 2))

        with blackjax.progress_bar(label="vmap") as state:
            (final_state, _) = jax.vmap(run)(keys, positions)
            jax.block_until_ready(final_state)

        chex.assert_shape(final_state.position, (2, 2))
        self.assertEqual(state.n_steps, 50)
        self.assertGreater(state.current_step, 0)

    def test_outermost_only(self):
        """Only the outermost scan (10 steps) is instrumented; the inner
        scan (5 steps, 50 total) must not double-fire the callback."""

        def inner_body(carry, x):
            return carry + x, carry

        def outer_body(carry, x):
            inner_final, _ = jax.lax.scan(inner_body, carry, jnp.arange(5))
            return inner_final, inner_final

        with blackjax.progress_bar(label="nested") as state:
            calls = []
            original_callback = state._step_callback

            def counting_callback(idx):
                calls.append(int(idx))
                original_callback(idx)

            state._step_callback = counting_callback

            final, _ = jax.lax.scan(outer_body, 0.0, jnp.arange(10))
            jax.block_until_ready(final)

        self.assertEqual(len(calls), 10)
        self.assertLessEqual(max(calls), 9)

    def test_output_file(self):
        """``output_file`` is written atomically during the run and removed
        on exit."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "bjx_progress.txt")

        def body(carry, x):
            return carry + x, carry

        with blackjax.progress_bar(
            label="file", output_file=path, print_rate=1
        ) as state:
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(20))
            jax.block_until_ready(final)

            self.assertTrue(os.path.exists(path))
            step, total = read_progress(path)
            self.assertEqual(total, state.n_steps)
            self.assertGreaterEqual(step, 0)
            self.assertLess(step, total)

        self.assertFalse(os.path.exists(path))

    def test_progress_reader_read_progress(self):
        """Unit test the standalone parse helper directly (no subprocess)."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "progress.txt")

        # Missing file -> None.
        self.assertIsNone(read_progress(path))

        with open(path, "w") as fh:
            fh.write("7 42")
        self.assertEqual(read_progress(path), (7, 42))

        # Malformed content -> None, not an exception.
        with open(path, "w") as fh:
            fh.write("not-a-number")
        self.assertIsNone(read_progress(path))

    def test_restoration(self):
        """After the context exits, ``jax.lax.scan`` is restored to the
        original, unpatched function and behaves normally."""
        with blackjax.progress_bar(label="restore-me"):
            final, _ = jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(5))
            jax.block_until_ready(final)

        self.assertIs(jax.lax.scan, _original_scan)

        # A plain scan after exit works exactly as jax.lax.scan normally does
        # -- no progress-bar machinery (and therefore no callback) is
        # reachable once jax.lax.scan is the unpatched function.
        final, ys = jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(5))
        np.testing.assert_allclose(final, 10.0)
        np.testing.assert_allclose(ys, jnp.array([0.0, 0.0, 1.0, 3.0, 6.0]))

    def test_sequential_scans_different_lengths(self):
        """Two sequential outermost scans of different lengths inside one
        context are each (re-)instrumented -- regression test for the
        multi-phase display bug (e.g. mclmc_find_L_and_step_size's several
        sequential warmup scans rendered with the first phase's total
        frozen for the rest of the run)."""
        calls = []

        def body(carry, x):
            return carry + x, carry

        with blackjax.progress_bar(label="sequential", print_rate=1) as state:
            original_callback = state._step_callback

            def counting_callback(idx):
                calls.append(int(idx))
                original_callback(idx)

            state._step_callback = counting_callback

            final1, _ = jax.lax.scan(body, 0.0, jnp.arange(30))
            final2, _ = jax.lax.scan(body, final1, jnp.arange(50))
            jax.block_until_ready(final2)

        self.assertEqual(len(calls), 30 + 50)
        self.assertEqual(calls.count(0), 2)  # both phases fired step 0
        self.assertEqual(calls[:30], list(range(30)))
        self.assertEqual(calls[30:], list(range(50)))
        self.assertEqual(state.n_steps, 50)

    def test_reverse_display_ascends(self):
        """``reverse=True`` must show an ascending display sequence ending
        at ``n - 1``, not a countdown that looks reset-to-zero on
        completion (adversarial-review finding #A4)."""
        n = 6
        seen = []

        def body(carry, x):
            return carry + x, carry

        with blackjax.progress_bar(label="reverse", print_rate=1) as state:
            original_callback = state._step_callback

            def recording_callback(idx):
                seen.append(int(idx))
                original_callback(idx)

            state._step_callback = recording_callback
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(float(n)), reverse=True)
            jax.block_until_ready(final)

        self.assertEqual(seen, list(range(n)))
        self.assertEqual(state.current_step, n - 1)

    def test_unwritable_output_file_completes_with_one_warning(self):
        """An ``output_file`` write failure must not crash the run --
        ``_step_callback`` is a total function -- and must warn exactly
        once, not once per remaining step (adversarial-review finding #B14,
        the sole crash path found)."""
        if os.geteuid() == 0:
            self.skipTest("running as root bypasses directory permissions")

        tmpdir = tempfile.mkdtemp()
        readonly_dir = os.path.join(tmpdir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, stat.S_IREAD | stat.S_IEXEC)
        bad_path = os.path.join(readonly_dir, "progress.txt")

        def body(carry, x):
            return carry + x, carry

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with blackjax.progress_bar(
                    label="perm", output_file=bad_path, print_rate=1
                ) as state:
                    final, _ = jax.lax.scan(body, 0.0, jnp.arange(50))
                    jax.block_until_ready(final)
            output_file_warnings = [
                w
                for w in caught
                if issubclass(w.category, UserWarning)
                and "output_file" in str(w.message)
            ]
            self.assertEqual(len(output_file_warnings), 1)
            self.assertEqual(state.current_step, 49)
            np.testing.assert_allclose(final, float(jnp.arange(50.0).sum()))
        finally:
            os.chmod(readonly_dir, stat.S_IRWXU)

    def test_print_rate_zero_no_crash(self):
        """``print_rate=0`` (an innocent typo) must not
        ``ZeroDivisionError`` inside the callback."""

        def body(carry, x):
            return carry + x, carry

        with blackjax.progress_bar(label="zero-rate", print_rate=0) as state:
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(30))
            jax.block_until_ready(final)

        self.assertEqual(state.current_step, 29)

    def test_step_callback_survives_promoted_warnings(self):
        """The never-raise invariant must hold even when the active
        warnings filter promotes ``UserWarning`` to an error (this
        project's own ``pytest.ini`` does): driving ``_step_callback``
        directly against an unwritable ``output_file`` must not raise, and
        ``output_file`` must still end up disabled (TL round-2 finding --
        the courtesy warning itself was escaping as the very crash item A
        exists to prevent)."""
        if os.geteuid() == 0:
            self.skipTest("running as root bypasses directory permissions")

        tmpdir = tempfile.mkdtemp()
        readonly_dir = os.path.join(tmpdir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, stat.S_IREAD | stat.S_IEXEC)
        bad_path = os.path.join(readonly_dir, "progress.txt")

        try:
            state = ProgressState(label="direct", print_rate=1, output_file=bad_path)
            state.n_steps = 10
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    state._step_callback(jnp.array(0))
                except Exception as e:  # pragma: no cover -- failure path
                    self.fail(
                        "_step_callback raised under a promoted warnings "
                        f"filter: {e!r}"
                    )
            self.assertIsNone(state.output_file)
        finally:
            os.chmod(readonly_dir, stat.S_IRWXU)

    def test_concurrent_first_enter_no_self_capture(self):
        """Two threads racing to be the FIRST ``progress_bar()`` entrant
        must never let one capture the OTHER's freshly-installed
        ``_patched_scan`` as ``_session_scan`` (which would self-recurse on
        every pass-through call). Repeated with a ``threading.Barrier`` so
        both threads hit ``__enter__`` at the same instant.

        M1 adaptation: self-capture guard is now jaxtap's responsibility.
        We verify jaxtap's own invariant: ``_jaxtap_ashell._session_scan``
        must not equal ``jaxtap._ashell._patched_scan`` inside either
        thread's context.  The semantics are identical to the #964 test --
        only which module owns the guard changes.
        """

        def body(carry, x):
            return carry + x, carry

        for i in range(50):
            barrier = threading.Barrier(2)
            captured = []
            results = {}

            def worker(label, n):
                barrier.wait(timeout=5)
                with blackjax.progress_bar(label=label) as state:
                    # Capture jaxtap's _session_scan (the #964 invariant now
                    # lives in jaxtap._ashell, not in blackjax.progress_bar).
                    captured.append(_jaxtap_ashell._session_scan)
                    f, _ = jax.lax.scan(body, 0.0, jnp.arange(n))
                    jax.block_until_ready(f)
                    results[label] = state.n_steps

            t1 = threading.Thread(target=worker, args=("r1", 4))
            t2 = threading.Thread(target=worker, args=("r2", 5))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            self.assertFalse(t1.is_alive(), f"iteration {i}: thread 1 hung")
            self.assertFalse(t2.is_alive(), f"iteration {i}: thread 2 hung")
            for c in captured:
                self.assertIsNot(
                    c, _patched_scan, f"iteration {i}: captured our own patch"
                )
            self.assertEqual(results.get("r1"), 4)
            self.assertEqual(results.get("r2"), 5)

        self.assertIs(jax.lax.scan, _original_scan)

    def test_owner_thread_routing_two_contexts(self):
        """With >=2 simultaneously active contexts, a scan is attributed
        only to its OWNER thread (the thread that called ``__enter__``); a
        true bystander thread owning neither context falls through with no
        bar rather than being misattributed (adversarial-review finding
        #B4)."""

        def body(carry, x):
            return carry + x, carry

        def run_scan(n):
            f, _ = jax.lax.scan(body, 0.0, jnp.arange(n))
            jax.block_until_ready(f)

        # -- owner-correct cell: each thread's own scan lands on its own
        # context, never on the other's. --
        ready1, ready2, done1 = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        results = {}

        def owner_a():
            with blackjax.progress_bar(label="A") as sa:
                results["a"] = sa
                ready1.set()
                ready2.wait(timeout=5)
                run_scan(31)
                done1.set()

        def owner_b():
            ready1.wait(timeout=5)
            with blackjax.progress_bar(label="B") as sb:
                results["b"] = sb
                ready2.set()
                done1.wait(timeout=5)
                run_scan(32)

        ta, tb = threading.Thread(target=owner_a), threading.Thread(target=owner_b)
        ta.start()
        tb.start()
        ta.join(timeout=10)
        tb.join(timeout=10)

        self.assertEqual(results["a"].n_steps, 31)
        self.assertEqual(results["b"].n_steps, 32)

        # -- bystander cell: a genuine third thread owning NEITHER active
        # context must not be misattributed to either. --
        ready1.clear()
        ready2.clear()
        done1.clear()
        results2 = {}

        def owner_a2():
            with blackjax.progress_bar(label="A2") as sa:
                results2["a"] = sa
                ready1.set()
                ready2.wait(timeout=5)
                bystander = threading.Thread(target=lambda: run_scan(41))
                bystander.start()
                bystander.join(timeout=5)
                done1.set()

        def owner_b2():
            ready1.wait(timeout=5)
            with blackjax.progress_bar(label="B2") as sb:
                results2["b"] = sb
                ready2.set()
                done1.wait(timeout=5)

        # Both A2 and B2 legitimately end this cell with zero attributed
        # steps (that is the point of the test -- the bystander's scan
        # must land on neither), which by design now also triggers the
        # zero-step UserWarning on each context's exit; suppress it here
        # since it's expected, not a signal something went wrong under this
        # project's filterwarnings=error.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ta2 = threading.Thread(target=owner_a2)
            tb2 = threading.Thread(target=owner_b2)
            ta2.start()
            tb2.start()
            ta2.join(timeout=10)
            tb2.join(timeout=10)

        self.assertEqual(results2["a"].n_steps, 0)
        self.assertEqual(results2["b"].n_steps, 0)

    def test_foreign_patch_chaining_and_restoration(self):
        """A third-party ``jax.lax.scan`` patch installed before our context
        keeps firing for scans run inside it (chaining), and is restored --
        not clobbered back to the true original -- on exit
        (adversarial-review finding #B8)."""
        foreign_calls = []
        real_original = _original_scan

        def foreign_patch(f, init, xs=None, length=None, **kwargs):
            foreign_calls.append(1)
            return real_original(f, init, xs=xs, length=length, **kwargs)

        def body(carry, x):
            return carry + x, carry

        jax.lax.scan = foreign_patch
        try:
            with blackjax.progress_bar(label="chain-test") as state:
                final, _ = jax.lax.scan(body, 0.0, jnp.arange(15))
                jax.block_until_ready(final)

            self.assertGreater(len(foreign_calls), 0)
            self.assertEqual(state.n_steps, 15)
            # Restored to the FOREIGN patch, not clobbered back to the true
            # original -- the foreign patch predates our session.
            self.assertIs(jax.lax.scan, foreign_patch)
        finally:
            jax.lax.scan = real_original

    def test_nonlifo_two_context_restore(self):
        """A enters, B enters, A exits first, B exits last --
        ``jax.lax.scan`` is restored to the true original once the LAST
        context exits, regardless of entry/exit order (adversarial-review
        finding #B8, non-LIFO exit-order case)."""

        def body(carry, x):
            return carry + x, carry

        cm_a = blackjax.progress_bar(label="A")
        cm_b = blackjax.progress_bar(label="B")
        cm_a.__enter__()
        cm_b.__enter__()
        try:
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(9))
            jax.block_until_ready(final)
        finally:
            # Both contexts are owned by this (the main) thread, so the
            # single scan above is attributed only to B (the most recently
            # entered of the two, per the owner-tie-break rule) -- A
            # legitimately sees zero steps and warns on exit; that warning
            # is expected here; it is not what this test is checking.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                cm_a.__exit__(None, None, None)
            self.assertIs(jax.lax.scan, _patched_scan)  # B is still active
            cm_b.__exit__(None, None, None)

        self.assertIs(jax.lax.scan, _original_scan)

    def test_zero_step_warns(self):
        """A context that never observes a scan step warns once, naming
        both possible root causes."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with blackjax.progress_bar(label="no-scan"):
                pass

        messages = [
            str(w.message) for w in caught if issubclass(w.category, UserWarning)
        ]
        self.assertTrue(any("no scan step was ever observed" in m for m in messages))

    def test_kwargs_passthrough(self):
        """``reverse=True`` and ``unroll=2`` are forwarded faithfully and
        still produce correct results."""

        def body(carry, x):
            return carry + x, carry

        xs = jnp.arange(6, dtype=jnp.float32)
        expected_final, expected_ys = jax.lax.scan(
            body, 0.0, xs, reverse=True, unroll=2
        )

        with blackjax.progress_bar(label="kwargs"):
            final, ys = jax.lax.scan(body, 0.0, xs, reverse=True, unroll=2)
            jax.block_until_ready(final)

        np.testing.assert_allclose(final, expected_final)
        np.testing.assert_allclose(ys, expected_ys)

    def test_compose_jaxtap_record_inside_progress_bar(self):
        """``blackjax.progress_bar()`` + user's ``jaxtap.record()`` simultaneously.

        Parity checklist item 6 (compose test).

        With >=2 jaxtap contexts active, attribution follows the
        innermost-owner-thread-wins rule from jaxtap's A-shell: scans run
        inside the user's ``tap.record()`` block route to the user's
        recorder, not to the progress bar.  Scans run outside the user's
        block (but inside ``progress_bar()``) continue to route to the
        progress bar.

        Semantics verdict [TESTED]: both consumers receive events from the
        scans attributed to them; no crash; no event corruption.  The
        progress bar does not update while the user's context is innermost
        -- documented boundary, not a bug (matches jaxtap's
        ``test_ashell_reentrant_contexts`` contract).
        """
        import jaxtap as tap

        def body(carry, x):
            return carry + x, carry

        with blackjax.progress_bar(label="outer") as pstate:
            # Scan 1: outside user's context → progress bar gets it.
            final1, _ = jax.lax.scan(body, 0.0, jnp.arange(10))
            jax.block_until_ready(final1)

            # User opens their own tap.record() inside progress_bar():
            # innermost-wins → user's recorder gets the scan, bar silent.
            with tap.record() as user_rec:
                final2, _ = jax.lax.scan(body, final1, jnp.arange(15))
                jax.block_until_ready(final2)

            # Scan 3: user's context exited → progress bar active again.
            final3, _ = jax.lax.scan(body, final2, jnp.arange(5))
            jax.block_until_ready(final3)

        # User's recorder received exactly the 15-step scan.
        user_scan_events = [e for e in user_rec.events if e.path.startswith("scan")]
        self.assertEqual(len(user_scan_events), 15)

        # Progress bar saw the 10-step and 5-step scans; n_steps = 5 (the
        # last scan attributed to it).  The 15-step scan was NOT routed here.
        self.assertEqual(pstate.n_steps, 5)

        # No crash; scan restored to the import-time original.
        self.assertIs(jax.lax.scan, _original_scan)


if __name__ == "__main__":
    absltest.main()
