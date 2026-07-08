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
"""Unit tests for the ``jax.lax.scan``-monkeypatching progress bar."""
import os
import tempfile

import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import blackjax
from blackjax.progress_bar import _original_scan
from blackjax.progress_reader import read_progress
from blackjax.util import run_inference_algorithm
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


if __name__ == "__main__":
    absltest.main()
