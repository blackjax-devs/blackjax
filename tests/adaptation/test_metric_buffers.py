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
"""Golden and contract tests for blackjax.adaptation.metric_buffers (D-layer).

Test strategy
-------------
Each test class follows the **frozen-inline-reference pattern**: the reference
implementation is embedded directly (not imported) so that any mutation to the
production module does not silently alias the golden.

Coverage plan
~~~~~~~~~~~~~
- **CGL-merge (Chan–Golub–LeVeque) exactness**: merged blocks == single-pass
  moments on the same data (f64, tight atol 1e-10).
- **Pop-oldest exactness**: merge of k-1 blocks equals recomputation from
  scratch on the same draws, for all valid k values.
- **reset_window equivalence**: ``push_split`` zeros the accumulator; a
  subsequent ``update`` restart is equivalent to accumulating from scratch.
- **Ensemble draw-axis split semantics**: all n_chains fold into one block
  per time-step, not one block per chain.
- **Diagonal-reference contract**: ``get_diag_reference`` equals
  ``diag(merged_M2) / max(count-1, 1)`` from the merged block.
- **Scan-carry shape stability**: jitting a multi-window scan asserts no
  recompilation beyond the initial trace.
- **requires_draws default-off**: default state carries no raw-draw ring;
  passing ``True`` raises ``NotImplementedError``.
- **f32 merge-accuracy golden**: run LAST (see class-level box-gate note);
  f32 Chan-merged moments vs f64 reference at d≈400, n≈64k.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from blackjax.adaptation.meads_adaptation import (
    _lrd_accumulator_init,
    _lrd_accumulator_update,
)
from blackjax.adaptation.metric_buffers import (
    AccumulatingSplitPopState,
    LateStartState,
    MomentBlock,
    accumulating_split_pop_buffer,
    cgl_merge_two,
    cgl_update_batch,
    diag_from_moment_block,
    ensemble_batch_buffer,
    late_start,
    merge_block_ring,
    reset_window_buffer,
)
from blackjax.adaptation.metric_estimators import sample_covariance_eigh_low_rank
from tests.fixtures import BlackJAXTest

# ---------------------------------------------------------------------------
# Frozen inline reference implementations (parity goldens)
# NOT imports — embedded to prevent silent aliasing.
# ---------------------------------------------------------------------------


def _ref_single_pass_moments(draws: np.ndarray):
    """Reference: single-pass mean and M2 via numpy (exact for comparison)."""
    n = draws.shape[0]
    mean = draws.mean(axis=0)
    centered = draws - mean
    m2 = centered.T @ centered  # (d, d)
    return float(n), mean, m2


def _ref_single_pass_diag_moments(draws: np.ndarray):
    """Reference: single-pass mean and diagonal M2 via numpy."""
    n = draws.shape[0]
    mean = draws.mean(axis=0)
    centered = draws - mean
    m2_diag = np.sum(centered**2, axis=0)  # (d,)
    return float(n), mean, m2_diag


def _ref_cgl_merge_two(na, mean_a, m2_a, nb, mean_b, m2_b):
    """Frozen CGL-merge reference (two pre-accumulated blocks, dense M2)."""
    n_ab = na + nb
    if n_ab == 0:
        return 0.0, np.zeros_like(mean_a), np.zeros_like(m2_a)
    delta = mean_b - mean_a
    mean_ab = mean_a + delta * (nb / n_ab)
    m2_ab = m2_a + m2_b + np.outer(delta, delta) * (na * nb / n_ab)
    return n_ab, mean_ab, m2_ab


def _ref_cgl_merge_ring(draws_list):
    """Frozen reference: CGL-merge a list of draw arrays (each (n_i, d))."""
    n_acc = 0.0
    mean_acc = None
    m2_acc = None
    for draws in draws_list:
        nb = float(draws.shape[0])
        mean_b = draws.mean(axis=0)
        centered_b = draws - mean_b
        m2_b = centered_b.T @ centered_b
        if n_acc == 0:
            n_acc, mean_acc, m2_acc = nb, mean_b.copy(), m2_b.copy()
        else:
            n_acc, mean_acc, m2_acc = _ref_cgl_merge_two(
                n_acc, mean_acc, m2_acc, nb, mean_b, m2_b
            )
    return n_acc, mean_acc, m2_acc


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_draws(key, n, d, scale=1.0):
    """Correlated Gaussian draws, shape (n, d)."""
    rho = 0.6
    corr = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
    L = np.linalg.cholesky(corr)
    z = jax.random.normal(key, (n, d))
    return np.array(z @ (scale * L).T)


def _make_block(n_f, mean, m2, dtype=np.float64):
    """Helper: wrap numpy arrays into a MomentBlock at the given dtype."""
    return MomentBlock(
        count=jnp.asarray(n_f, dtype=dtype),
        mean=jnp.asarray(mean, dtype=dtype),
        m2=jnp.asarray(m2, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# 1. Chan-merge exactness
# ---------------------------------------------------------------------------


def _tols_for_dtype(dtype):
    """Return (atol, rtol) for parity checks at a given dtype.

    f64 (with ``jax_enable_x64=True``): tight absolute tolerance 1e-9.
    f32 (JAX default): relative tolerance 1e-4; atol 0.  Relative tolerance
    is more robust than absolute for M2 values (which scale with n × variance).

    The rtol=1e-4 bound accounts for values near zero: when the true mean is
    O(1e-4), one extra f32 rounding step produces absolute error O(eps·|sum|)
    whose relative magnitude vs the mean can reach ~1e-4.  A tighter rtol=1e-5
    falsely rejects valid f32 arithmetic at that scale.
    """
    if np.dtype(dtype) == np.float64:
        return 1e-9, 0.0  # atol, rtol
    return 0.0, 1e-4  # atol, rtol for f32


def _assert_allclose_dtype(actual, desired, dtype, err_msg=""):
    """allclose wrapper applying dtype-appropriate tolerances."""
    atol, rtol = _tols_for_dtype(dtype)
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol, err_msg=err_msg)


class CGLMergeTwoTest(BlackJAXTest):
    """CGL-merge (Chan–Golub–LeVeque) of two pre-accumulated blocks == single-pass on the union.

    Tests run at the dtype JAX was launched with.  When ``jax_enable_x64=True``
    inputs are cast to f64 and tolerance is 1e-9; otherwise f32 at 1e-5.
    The f32 case does not invalidate the golden — both the merge AND the
    reference run the same f32 arithmetic, so agreement to 1e-5 is tight.
    """

    def _compute_dtype(self):
        """f64 if x64 enabled, else f32 — consistent with JAX's behaviour."""
        return np.float64 if jax.config.jax_enable_x64 else np.float32

    @parameterized.named_parameters(
        {"testcase_name": "d2_n5_n8", "d": 2, "n_a": 5, "n_b": 8},
        {"testcase_name": "d10_n20_n30", "d": 10, "n_a": 20, "n_b": 30},
        {"testcase_name": "d50_n100_n200", "d": 50, "n_a": 100, "n_b": 200},
    )
    def test_merge_equals_single_pass(self, d, n_a, n_b):
        """cgl_merge_two(A, B) matches single-pass moments on A∪B.

        When run with ``jax_enable_x64=True``, tolerance is 1e-9 (f64 tight).
        Default f32 run: tolerance 1e-5.
        """
        dtype = self._compute_dtype()
        key_a, key_b = jax.random.split(self.next_key(), 2)
        draws_a = _make_draws(key_a, n_a, d).astype(dtype)
        draws_b = _make_draws(key_b, n_b, d).astype(dtype)

        # Reference: single-pass on the concatenated union (numpy, same dtype)
        all_draws = np.concatenate([draws_a, draws_b], axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

        na_f, mean_a, m2_a = _ref_single_pass_moments(draws_a)
        nb_f, mean_b, m2_b = _ref_single_pass_moments(draws_b)

        block_a = _make_block(na_f, mean_a, m2_a, dtype=dtype)
        block_b = _make_block(nb_f, mean_b, m2_b, dtype=dtype)

        merged = cgl_merge_two(block_a, block_b)

        self.assertAlmostEqual(float(merged.count), ref_n, places=5)
        _assert_allclose_dtype(
            np.array(merged.mean),
            ref_mean.astype(dtype),
            dtype,
            err_msg="merged mean != single-pass mean",
        )
        _assert_allclose_dtype(
            np.array(merged.m2),
            ref_m2.astype(dtype),
            dtype,
            err_msg="merged M2 != single-pass M2",
        )

    def test_merge_with_empty_block(self):
        """Merging with an empty block (count=0) returns the non-empty block."""
        dtype = self._compute_dtype()
        key = self.next_key()
        d, n = 8, 20
        draws = _make_draws(key, n, d).astype(dtype)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws)

        block = _make_block(ref_n, ref_mean, ref_m2, dtype=dtype)
        empty = MomentBlock(
            count=jnp.asarray(0.0, dtype=jnp.asarray(ref_n).dtype),
            mean=jnp.zeros((d,), dtype=jnp.asarray(ref_mean).dtype),
            m2=jnp.zeros((d, d), dtype=jnp.asarray(ref_m2).dtype),
        )

        # empty ∪ block
        merged_1 = cgl_merge_two(empty, block)
        _assert_allclose_dtype(np.array(merged_1.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged_1.m2), ref_m2.astype(dtype), dtype)
        self.assertAlmostEqual(float(merged_1.count), ref_n, places=5)

        # block ∪ empty
        merged_2 = cgl_merge_two(block, empty)
        _assert_allclose_dtype(np.array(merged_2.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged_2.m2), ref_m2.astype(dtype), dtype)

    def test_diagonal_variant(self):
        """chan_merge_two works for diagonal M2 (1-D m2 field)."""
        dtype = self._compute_dtype()
        key_a, key_b = jax.random.split(self.next_key(), 2)
        d, n_a, n_b = 15, 30, 50
        draws_a = _make_draws(key_a, n_a, d).astype(dtype)
        draws_b = _make_draws(key_b, n_b, d).astype(dtype)

        na_f, mean_a, m2_a_diag = _ref_single_pass_diag_moments(draws_a)
        nb_f, mean_b, m2_b_diag = _ref_single_pass_diag_moments(draws_b)

        block_a = MomentBlock(
            count=jnp.asarray(na_f, dtype=dtype),
            mean=jnp.asarray(mean_a, dtype=dtype),
            m2=jnp.asarray(m2_a_diag, dtype=dtype),
        )
        block_b = MomentBlock(
            count=jnp.asarray(nb_f, dtype=dtype),
            mean=jnp.asarray(mean_b, dtype=dtype),
            m2=jnp.asarray(m2_b_diag, dtype=dtype),
        )
        merged = cgl_merge_two(block_a, block_b)

        all_draws = np.concatenate([draws_a, draws_b], axis=0)
        ref_n, ref_mean, ref_m2_diag = _ref_single_pass_diag_moments(all_draws)

        _assert_allclose_dtype(np.array(merged.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged.m2), ref_m2_diag.astype(dtype), dtype)

    def test_merge_equals_single_pass_x64(self):
        """cgl_merge_two agrees with single-pass moments at f64 tight tolerance.

        Explicitly enables x64 via context manager so that the atol=1e-9 path
        exercises even when JAX is running in f32-default mode (the default CI
        environment).
        """
        with jax.enable_x64():
            dtype = np.float64
            d, n_a, n_b = 10, 20, 30
            key_a, key_b = jax.random.split(self.next_key(), 2)
            draws_a = _make_draws(key_a, n_a, d).astype(dtype)
            draws_b = _make_draws(key_b, n_b, d).astype(dtype)

            all_draws = np.concatenate([draws_a, draws_b], axis=0)
            ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

            na_f, mean_a, m2_a = _ref_single_pass_moments(draws_a)
            nb_f, mean_b, m2_b = _ref_single_pass_moments(draws_b)
            block_a = _make_block(na_f, mean_a, m2_a, dtype=dtype)
            block_b = _make_block(nb_f, mean_b, m2_b, dtype=dtype)
            merged = cgl_merge_two(block_a, block_b)

            np.testing.assert_allclose(
                np.array(merged.mean),
                ref_mean.astype(dtype),
                atol=1e-9,
                rtol=0.0,
                err_msg="x64: merged mean != single-pass mean",
            )
            np.testing.assert_allclose(
                np.array(merged.m2),
                ref_m2.astype(dtype),
                atol=1e-9,
                rtol=0.0,
                err_msg="x64: merged M2 != single-pass M2",
            )


class CGLUpdateBatchTest(BlackJAXTest):
    """cgl_update_batch(block, batch) == cgl_merge_two(block, single_pass(batch))."""

    def test_matches_cgl_merge_two(self):
        """cgl_update_batch equals single-pass on A∪B at appropriate dtype tolerance."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        key_a, key_b = jax.random.split(self.next_key(), 2)
        d, n_a, n_b = 12, 25, 40
        draws_a = _make_draws(key_a, n_a, d).astype(dtype)
        draws_b = _make_draws(key_b, n_b, d).astype(dtype)

        na_f, mean_a, m2_a = _ref_single_pass_moments(draws_a)
        block_a = _make_block(na_f, mean_a, m2_a, dtype=dtype)

        updated = cgl_update_batch(block_a, jnp.asarray(draws_b))

        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(
            np.concatenate([draws_a, draws_b], axis=0)
        )

        _assert_allclose_dtype(np.array(updated.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(updated.m2), ref_m2.astype(dtype), dtype)
        self.assertAlmostEqual(float(updated.count), ref_n, places=5)

    def test_single_draw_shape(self):
        """A single (1, d) draw is accepted and produces count=1."""
        key = self.next_key()
        d = 7
        draw = jax.random.normal(key, (d,))

        empty = MomentBlock(
            count=jnp.zeros(()),
            mean=jnp.zeros((d,)),
            m2=jnp.zeros((d, d)),
        )
        updated = cgl_update_batch(empty, draw[None, :])  # explicit (1, d)
        self.assertAlmostEqual(float(updated.count), 1.0)
        np.testing.assert_allclose(np.array(updated.mean), np.array(draw), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. merge_block_ring exactness
# ---------------------------------------------------------------------------


class MergeBlockRingTest(BlackJAXTest):
    """merge_block_ring(counts, means, m2s) == single-pass on all draws.

    Tests at the dtype JAX was launched with; tolerance matches precision.
    When run with ``jax_enable_x64=True`` the tolerance is 1e-9 (f64 tight),
    meeting the Chan-merge exactness golden requirement.
    """

    @parameterized.named_parameters(
        {"testcase_name": "k2_d5", "k": 2, "d": 5, "n_per_block": 20},
        {"testcase_name": "k4_d10", "k": 4, "d": 10, "n_per_block": 30},
        {"testcase_name": "k8_d20", "k": 8, "d": 20, "n_per_block": 50},
    )
    def test_ring_merge_equals_single_pass(self, k, d, n_per_block):
        """merge_block_ring == single-pass moments on all draws."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        keys = jax.random.split(self.next_key(), k)
        draws_list = [
            _make_draws(keys[i], n_per_block, d).astype(dtype) for i in range(k)
        ]

        counts = []
        means = []
        m2s = []
        for draws in draws_list:
            n, mean, m2 = _ref_single_pass_moments(draws)
            counts.append(n)
            means.append(mean)
            m2s.append(m2)

        counts_arr = jnp.asarray(counts, dtype=dtype)
        means_arr = jnp.asarray(means, dtype=dtype)
        m2s_arr = jnp.asarray(m2s, dtype=dtype)

        merged = merge_block_ring(counts_arr, means_arr, m2s_arr)

        all_draws = np.concatenate(draws_list, axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

        _assert_allclose_dtype(np.array(merged.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged.m2), ref_m2.astype(dtype), dtype)
        self.assertAlmostEqual(float(merged.count), ref_n, places=5)

    def test_ring_with_empty_slots(self):
        """Empty slots (count=0) are transparent to the merge."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        k, d, n = 4, 6, 25
        key = self.next_key()
        draws = _make_draws(key, n, d).astype(dtype)
        n_f, mean_, m2_ = _ref_single_pass_moments(draws)

        # k-1 empty slots; only the first slot carries data
        empty_counts = [0.0] * (k - 1)
        counts = jnp.asarray([n_f] + empty_counts, dtype=dtype)
        means = jnp.stack(
            [jnp.asarray(mean_, dtype=dtype)] + [jnp.zeros((d,), dtype=dtype)] * (k - 1)
        )
        m2s = jnp.stack(
            [jnp.asarray(m2_, dtype=dtype)] + [jnp.zeros((d, d), dtype=dtype)] * (k - 1)
        )

        merged = merge_block_ring(counts, means, m2s)

        _assert_allclose_dtype(np.array(merged.mean), mean_.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged.m2), m2_.astype(dtype), dtype)
        self.assertAlmostEqual(float(merged.count), n_f, places=5)

    def test_ring_merge_equals_single_pass_x64(self):
        """merge_block_ring agrees with single-pass moments at f64 tight tolerance.

        Explicitly enables x64 via context manager so that atol=1e-9 exercises
        even in f32-default CI.
        """
        with jax.enable_x64():
            dtype = np.float64
            k, d, n_per_block = 4, 10, 30
            keys = jax.random.split(self.next_key(), k)
            draws_list = [
                _make_draws(keys[i], n_per_block, d).astype(dtype) for i in range(k)
            ]

            counts, means, m2s = [], [], []
            for draws in draws_list:
                n, mean, m2 = _ref_single_pass_moments(draws)
                counts.append(n)
                means.append(mean)
                m2s.append(m2)

            counts_arr = jnp.asarray(counts, dtype=dtype)
            means_arr = jnp.asarray(means, dtype=dtype)
            m2s_arr = jnp.asarray(m2s, dtype=dtype)
            merged = merge_block_ring(counts_arr, means_arr, m2s_arr)

            all_draws = np.concatenate(draws_list, axis=0)
            ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

            np.testing.assert_allclose(
                np.array(merged.mean),
                ref_mean.astype(dtype),
                atol=1e-9,
                rtol=0.0,
                err_msg="x64: ring merge mean != single-pass mean",
            )
            np.testing.assert_allclose(
                np.array(merged.m2),
                ref_m2.astype(dtype),
                atol=1e-9,
                rtol=0.0,
                err_msg="x64: ring merge M2 != single-pass M2",
            )


# ---------------------------------------------------------------------------
# 3. reset_window policy
# ---------------------------------------------------------------------------


class ResetWindowBufferTest(BlackJAXTest):
    """Policy 1: reset_window_buffer correctness."""

    def _run_accumulation(self, init, update, fns_rest, d, n, key):
        """Accumulate n draws one at a time; return state."""
        push_split, get_moments, get_support, get_diag_reference = fns_rest
        state = init()
        draws = _make_draws(key, n, d)
        for i in range(n):
            state = update(state, jnp.asarray(draws[i]))  # (d,) draws
        return state, draws

    def test_moments_match_single_pass(self):
        """Accumulated moments match single-pass reference."""
        d, n = 8, 40
        key = self.next_key()
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d)
        state, draws = self._run_accumulation(
            init,
            update,
            (push_split, get_moments, get_support, get_diag_ref),
            d,
            n,
            key,
        )

        block = get_moments(state)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2, rtol=1e-4)
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_push_split_zeros_accumulator(self):
        """push_split zeroes the accumulator (hard reset)."""
        d, n = 5, 20
        key = self.next_key()
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d)
        state, _ = self._run_accumulation(
            init,
            update,
            (push_split, get_moments, get_support, get_diag_ref),
            d,
            n,
            key,
        )

        reset_state = push_split(state)
        block = get_moments(reset_state)

        self.assertAlmostEqual(float(block.count), 0.0)
        np.testing.assert_array_equal(np.array(block.mean), np.zeros((d,)))
        np.testing.assert_array_equal(np.array(block.m2), np.zeros((d, d)))

    def test_restart_after_reset_matches_fresh_accumulation(self):
        """Accumulating after a reset is equivalent to starting fresh."""
        d, n = 10, 30
        key1, key2 = jax.random.split(self.next_key(), 2)

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d)

        # Accumulate window 1, reset, accumulate window 2
        state = init()
        draws1 = _make_draws(key1, n, d)
        for i in range(n):
            state = update(state, jnp.asarray(draws1[i]))
        state = push_split(state)  # reset

        draws2 = _make_draws(key2, n, d)
        for i in range(n):
            state = update(state, jnp.asarray(draws2[i]))
        block = get_moments(state)

        # Fresh accumulation on draws2 only
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws2)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2, rtol=1e-4)
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_requires_draws_false_no_draw_ring(self):
        """Default state carries no raw-draw ring (opt-in capability, off by default).

        reset_window_buffer is a thin wrapper over the k=1 split-pop path, so
        the returned state is AccumulatingSplitPopState (4 fields).
        """
        d = 6
        init, _, _, _, _, _ = reset_window_buffer(d, requires_draws=False)
        state = init()
        # k=1 split-pop path: AccumulatingSplitPopState with 4 fields
        self.assertIsInstance(state, AccumulatingSplitPopState)
        self.assertEqual(len(state), 4)  # counts, means, m2s, write_pos

    def test_requires_draws_true_raises(self):
        """requires_draws=True raises NotImplementedError (raw-draw ring is opt-in, unimplemented)."""
        with self.assertRaises(NotImplementedError):
            reset_window_buffer(5, requires_draws=True)

    def test_diagonal_variant(self):
        """diagonal=True gives correct per-coordinate variance."""
        d, n = 12, 50
        key = self.next_key()
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d, diagonal=True)
        state = init()
        draws = _make_draws(key, n, d)
        for i in range(n):
            state = update(state, jnp.asarray(draws[i]))

        block = get_moments(state)
        ref_n, ref_mean, ref_m2_diag = _ref_single_pass_diag_moments(draws)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2_diag, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. accumulating_split_pop policy
# ---------------------------------------------------------------------------


class AccumulatingSplitPopTest(BlackJAXTest):
    """Policy 2: accumulating_split_pop_buffer correctness."""

    @parameterized.named_parameters(
        {"testcase_name": "k2_d5_n10", "k": 2, "d": 5, "n_per_split": 10},
        {"testcase_name": "k4_d8_n20", "k": 4, "d": 8, "n_per_split": 20},
        {"testcase_name": "k6_d12_n30", "k": 6, "d": 12, "n_per_split": 30},
    )
    def test_merge_after_k_splits_equals_all_draws(self, k, d, n_per_split):
        """After filling k splits, merged moments == single-pass on all draws."""
        keys = jax.random.split(self.next_key(), k)
        draws_list = [_make_draws(keys[i], n_per_split, d) for i in range(k)]

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = accumulating_split_pop_buffer(d, k)
        state = init()

        for split_idx in range(k):
            draws = draws_list[split_idx]
            for i in range(n_per_split):
                state = update(state, jnp.asarray(draws[i]))
            if split_idx < k - 1:
                state = push_split(state)

        block = get_moments(state)
        all_draws = np.concatenate(draws_list, axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2, rtol=1e-4)
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    @parameterized.named_parameters(
        # k=3 single-wrap: one extra split causes first block to be dropped
        {
            "testcase_name": "k3_d8_n20_nwraps1",
            "k": 3,
            "d": 8,
            "n_per_split": 20,
            "n_extra_wraps": 1,
        },
        # k=4 single-wrap: catches stride-2 ring-pointer aliasing invisible at k=3
        {
            "testcase_name": "k4_d6_n15_nwraps1",
            "k": 4,
            "d": 6,
            "n_per_split": 15,
            "n_extra_wraps": 1,
        },
        # k=4 double-wrap (n_splits = 2k+1): ring wraps twice; oldest is dropped twice
        {
            "testcase_name": "k4_d6_n15_nwraps2",
            "k": 4,
            "d": 6,
            "n_per_split": 15,
            "n_extra_wraps": 2,
        },
        # k=5 double-wrap
        {
            "testcase_name": "k5_d5_n10_nwraps2",
            "k": 5,
            "d": 5,
            "n_per_split": 10,
            "n_extra_wraps": 2,
        },
    )
    def test_pop_oldest_exactness(self, k, d, n_per_split, n_extra_wraps):
        """Merge of the k retained blocks == recomputation from scratch after multi-wrap.

        After n_extra_wraps complete ring passes, the retained window is the LAST
        k splits; older splits have been overwritten.  We verify that the buffer's
        merged moments exactly match single-pass recomputation on those k splits.

        Parameterized at k=4 (catches stride-2 ring-pointer aliasing invisible at k=3
        due to mod-3 aliasing) and with n_extra_wraps=2 (multi-wrap exercise).
        """
        n_splits_total = (
            k * n_extra_wraps + 1
        )  # one more than a full n_extra_wraps passes

        keys = jax.random.split(self.next_key(), n_splits_total)
        draws_list = [
            _make_draws(keys[i], n_per_split, d) for i in range(n_splits_total)
        ]

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = accumulating_split_pop_buffer(d, k)
        state = init()

        for split_idx in range(n_splits_total):
            draws = draws_list[split_idx]
            for i in range(n_per_split):
                state = update(state, jnp.asarray(draws[i]))
            if split_idx < n_splits_total - 1:
                state = push_split(state)

        block = get_moments(state)

        # After n_splits_total splits the ring retains exactly the last k splits.
        retained = draws_list[-k:]
        remaining_draws = np.concatenate(retained, axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(remaining_draws)

        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="pop-oldest: merged mean != recomputation from scratch",
        )
        np.testing.assert_allclose(
            np.array(block.m2),
            ref_m2,
            rtol=1e-4,
            err_msg="pop-oldest: merged M2 != recomputation from scratch",
        )
        self.assertAlmostEqual(
            float(block.count),
            ref_n,
            places=5,
            msg="pop-oldest: total count != recomputation count",
        )

    def test_support_reports_correct_totals(self):
        """get_support returns (total_count, per_block_counts) correctly."""
        k, d, n = 3, 5, 15
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = accumulating_split_pop_buffer(d, k)
        state = init()
        key = self.next_key()

        # Fill first split
        draws = _make_draws(key, n, d)
        for i in range(n):
            state = update(state, jnp.asarray(draws[i]))

        total, per_block = get_support(state)
        self.assertAlmostEqual(float(total), float(n), places=5)
        self.assertAlmostEqual(float(per_block[0]), float(n), places=5)  # write_pos=0
        # remaining slots are empty
        for j in range(1, k):
            self.assertAlmostEqual(float(per_block[j]), 0.0, places=8)

    def test_requires_draws_false_no_draw_ring(self):
        """Default state has no raw-draw ring (raw-draw ring is opt-in, off by default).

        num_valid was removed: it is recomputable as jnp.sum(counts > 0) and
        tracking it as a separate field caused stale values under consecutive
        empty pushes.  State now has exactly 4 fields.
        """
        d, k = 5, 3
        init, _, _, _, _, _ = accumulating_split_pop_buffer(d, k, requires_draws=False)
        state = init()
        self.assertIsInstance(state, AccumulatingSplitPopState)
        self.assertEqual(len(state), 4)  # counts, means, m2s, write_pos (no num_valid)

    def test_requires_draws_true_raises(self):
        """requires_draws=True raises NotImplementedError (raw-draw ring is opt-in, unimplemented)."""
        with self.assertRaises(NotImplementedError):
            accumulating_split_pop_buffer(5, 3, requires_draws=True)


# ---------------------------------------------------------------------------
# 5. ensemble_batch policy (draw-axis split semantics)
# ---------------------------------------------------------------------------


class EnsembleBatchBufferTest(BlackJAXTest):
    """Policy 3: ensemble_batch_buffer correctness + draw-axis split contract."""

    def test_ensemble_chains_fold_into_one_block(self):
        """All chains fold into the active block, not one block per chain.

        After one ensemble update with n_chains chains, there is exactly ONE
        block with count == n_chains (not n_chains blocks each with count 1).
        """
        d, n_chains, k = 8, 16, 4
        key = self.next_key()
        batch = jnp.asarray(_make_draws(key, n_chains, d))

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = ensemble_batch_buffer(d, n_chains, k)
        state = init()
        state = update(state, batch)  # (n_chains, d) — all chains fold together

        total, per_block = get_support(state)
        # One block should have count == n_chains; all others zero
        self.assertAlmostEqual(float(total), float(n_chains), places=5)
        # Active block is at write_pos=0 initially
        non_zero = [float(c) for c in per_block if float(c) > 0]
        self.assertEqual(len(non_zero), 1, msg="more than one block got data")
        self.assertAlmostEqual(non_zero[0], float(n_chains), places=5)

    def test_ensemble_split_is_time_axis_not_chain_partition(self):
        """push_split creates a new time block, not a chain-subset block.

        Two time-steps of ensemble updates → two blocks; each block combines
        all chains.  The merge == single-pass on both time-steps' chains.
        """
        d, n_chains, k = 6, 8, 4
        key1, key2 = jax.random.split(self.next_key(), 2)
        batch1 = jnp.asarray(_make_draws(key1, n_chains, d))
        batch2 = jnp.asarray(_make_draws(key2, n_chains, d))

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = ensemble_batch_buffer(d, n_chains, k)
        state = init()
        state = update(state, batch1)  # time step 1
        state = push_split(state)  # end of split 1, new time block starts
        state = update(state, batch2)  # time step 2

        block = get_moments(state)
        all_draws = np.concatenate([np.array(batch1), np.array(batch2)], axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_draws)

        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="merge not equal to all chains concat",
        )
        np.testing.assert_allclose(
            np.array(block.m2),
            ref_m2,
            rtol=1e-4,
            err_msg="M2 not equal to all chains concat",
        )
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_invalid_n_chains_raises(self):
        """n_chains < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            ensemble_batch_buffer(5, 0, 3)


# ---------------------------------------------------------------------------
# 5b. Exact parity: ensemble_batch_buffer == _lrd_accumulator_update under x64
# ---------------------------------------------------------------------------


class EnsembleMeadsParityX64Test(BlackJAXTest):
    """Under x64, ensemble_batch_buffer == _lrd_accumulator_update at exact (0.0) parity.

    Pinned after the dtype bug fix: with count/counts using the ambient float
    dtype (not hardcoded float32), the Chan merge weights are computed at f64
    precision, giving bit-identical results with the MEADS in-tree accumulator
    on the same data.  This test is the canary — if it fails, the dtype bug
    has been reintroduced.
    """

    def test_exact_parity_x64(self):
        """ensemble_batch_buffer == _lrd_accumulator_update, bit-exact under x64."""
        with jax.enable_x64():
            d, n_chains, n_steps = 8, 12, 20
            key = self.next_key()
            keys = jax.random.split(key, n_steps)
            # Cast to f64 inside x64 context so JAX promotes arithmetic to f64.
            batches = [
                jax.random.normal(keys[i], (n_chains, d)).astype(jnp.float64)
                for i in range(n_steps)
            ]

            # ensemble_batch_buffer with k=1 (single accumulating block, no ring)
            init, update, _, get_moments, _, _ = ensemble_batch_buffer(d, n_chains, k=1)
            state = init()
            for batch in batches:
                state = update(state, batch)
            block = get_moments(state)

            # MEADS in-tree accumulator (reference implementation)
            acc = _lrd_accumulator_init(d)
            for batch in batches:
                acc = _lrd_accumulator_update(acc, batch)

            # Bit-exact parity: 0.0 difference, not just small difference
            np.testing.assert_array_equal(
                np.array(block.count),
                np.array(acc.count),
                err_msg="count differs: dtype bug reintroduced?",
            )
            np.testing.assert_array_equal(
                np.array(block.mean),
                np.array(acc.mean),
                err_msg="mean differs: Chan weight precision mismatch",
            )
            np.testing.assert_array_equal(
                np.array(block.m2),
                np.array(acc.m2),
                err_msg="M2 differs: Chan weight precision mismatch",
            )


# ---------------------------------------------------------------------------
# 6. Diagonal-reference contract
# ---------------------------------------------------------------------------


class DiagReferenceTest(BlackJAXTest):
    """get_diag_reference == diag(merged M2) / max(count-1, 1)."""

    def test_diag_reference_equals_bessel_corrected_diag(self):
        """diag_reference matches manual Bessel-corrected diagonal."""
        d, n, k = 10, 50, 3
        keys = jax.random.split(self.next_key(), k)
        draws_list = [_make_draws(keys[i], n, d) for i in range(k)]

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = accumulating_split_pop_buffer(d, k)
        state = init()

        for split_idx in range(k):
            for i in range(n):
                state = update(state, jnp.asarray(draws_list[split_idx][i]))
            if split_idx < k - 1:
                state = push_split(state)

        diag_ref = get_diag_ref(state)
        block = get_moments(state)

        # Manual: diag(merged_M2) / max(count - 1, 1)
        merged_m2_diag = np.diag(np.array(block.m2))
        expected = merged_m2_diag / max(float(block.count) - 1.0, 1.0)

        np.testing.assert_allclose(
            np.array(diag_ref),
            expected,
            rtol=1e-4,
            err_msg="diag_reference != diag(M2)/max(n-1,1)",
        )

    def test_diag_from_moment_block_matches_numpy_var(self):
        """diag_from_moment_block == np.var(draws, ddof=1) (per-coordinate)."""
        d, n = 15, 100
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        key = self.next_key()
        draws = _make_draws(key, n, d)
        n_f, mean_, m2_ = _ref_single_pass_moments(draws)
        block = _make_block(n_f, mean_, m2_, dtype=dtype)

        diag = diag_from_moment_block(block)
        expected = np.var(draws, axis=0, ddof=1)  # Bessel-corrected

        np.testing.assert_allclose(
            np.array(diag),
            expected,
            rtol=1e-4,
            err_msg="diag_from_moment_block != var(draws, ddof=1)",
        )

    def test_diag_reference_from_reset_window(self):
        """get_diag_reference works for reset_window_buffer (k=1 path)."""
        d, n = 8, 30
        key = self.next_key()
        draws = _make_draws(key, n, d)

        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d)
        state = init()
        for i in range(n):
            state = update(state, jnp.asarray(draws[i]))

        diag_ref = get_diag_ref(state)
        expected = np.var(draws, axis=0, ddof=1)

        np.testing.assert_allclose(np.array(diag_ref), expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# 7. late_start composability
# ---------------------------------------------------------------------------


class LateStartTest(BlackJAXTest):
    """Policy 4: late_start offset wrapper correctness."""

    def test_skips_first_offset_steps(self):
        """The first offset_steps draws are skipped; only draws after contribute."""
        d, n, offset = 8, 50, 10
        key = self.next_key()
        draws = _make_draws(key, n, d)

        inner_fns = reset_window_buffer(d)
        (
            ls_init,
            ls_update,
            ls_push,
            ls_get_moments,
            ls_get_support,
            ls_get_diag,
        ) = late_start(inner_fns, offset_steps=offset)
        state = ls_init()

        for i in range(n):
            state = ls_update(state, jnp.asarray(draws[i]))

        block = ls_get_moments(state)
        # Only draws[offset:] should be in the block
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws[offset:])

        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="late_start: mean includes skipped draws",
        )
        np.testing.assert_allclose(
            np.array(block.m2),
            ref_m2,
            rtol=1e-4,
            err_msg="late_start: M2 includes skipped draws",
        )
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_wraps_accumulating_split_pop(self):
        """late_start composes correctly with accumulating_split_pop_buffer."""
        d, k, n_per_split, offset = 6, 3, 30, 5
        keys = jax.random.split(self.next_key(), k)
        draws_list = [_make_draws(keys[i], n_per_split, d) for i in range(k)]

        inner_fns = accumulating_split_pop_buffer(d, k)
        (
            ls_init,
            ls_update,
            ls_push,
            ls_get_moments,
            ls_get_support,
            ls_get_diag,
        ) = late_start(inner_fns, offset_steps=offset)
        state = ls_init()

        # Accumulate k splits, each with a late-start of `offset` draws
        for split_idx in range(k):
            draws = draws_list[split_idx]
            for i in range(n_per_split):
                state = ls_update(state, jnp.asarray(draws[i]))
            if split_idx < k - 1:
                state = ls_push(state)

        block = ls_get_moments(state)

        # Reference: for each split, only draws[offset:] are counted
        valid_draws = [d_[offset:] for d_ in draws_list]
        all_valid = np.concatenate(valid_draws, axis=0)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(all_valid)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2, rtol=1e-4)
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_zero_offset(self):
        """offset_steps=0 == no skip at all."""
        d, n = 7, 30
        key = self.next_key()
        draws = _make_draws(key, n, d)

        inner_fns = reset_window_buffer(d)
        (
            ls_init,
            ls_update,
            ls_push,
            ls_get_moments,
            ls_get_support,
            ls_get_diag,
        ) = late_start(inner_fns, offset_steps=0)
        state = ls_init()
        for i in range(n):
            state = ls_update(state, jnp.asarray(draws[i]))

        block = ls_get_moments(state)
        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws)

        np.testing.assert_allclose(np.array(block.mean), ref_mean, rtol=1e-4)
        np.testing.assert_allclose(np.array(block.m2), ref_m2, rtol=1e-4)
        self.assertAlmostEqual(float(block.count), ref_n, places=5)

    def test_state_is_late_start_state(self):
        """State type is LateStartState with inner + num_skipped."""
        d = 5
        inner_fns = reset_window_buffer(d)
        ls_init, _, _, _, _, _ = late_start(inner_fns, offset_steps=10)
        state = ls_init()
        self.assertIsInstance(state, LateStartState)
        self.assertEqual(len(state), 2)  # inner, num_skipped


# ---------------------------------------------------------------------------
# 8. Scan-carry shape stability
# ---------------------------------------------------------------------------


class ScanCarryShapeTest(BlackJAXTest):
    """Fixed-shape state: jitting a multi-window scan doesn't recompile."""

    def test_scan_stable_reset_window(self):
        """Multi-step scan on reset_window is jit-compiled once."""
        d, n_steps = 8, 20
        key = self.next_key()
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = reset_window_buffer(d)
        draws = jnp.asarray(_make_draws(key, n_steps, d))

        @jax.jit
        def run_scan(init_state, draws):
            def step(state, draw):
                return update(state, draw), None

            final_state, _ = jax.lax.scan(step, init_state, draws)
            return final_state

        # Run twice; if the state has static shapes, no recompile (no XlaRuntimeError).
        state1 = run_scan(init(), draws)
        state2 = run_scan(init(), draws)

        block1 = get_moments(state1)
        block2 = get_moments(state2)
        np.testing.assert_allclose(
            np.array(block1.mean),
            np.array(block2.mean),
            atol=1e-10,
            err_msg="scan results differ between two identical runs",
        )

    def test_scan_stable_accumulating_split_pop(self):
        """Multi-step scan on accumulating_split_pop with jit runs without recompile."""
        d, k, n_steps = 6, 3, 24
        key = self.next_key()
        (
            init,
            update,
            push_split,
            get_moments,
            get_support,
            get_diag_ref,
        ) = accumulating_split_pop_buffer(d, k)
        draws = jnp.asarray(_make_draws(key, n_steps, d))

        # Scan over updates (no push_split inside scan — push_split is a Python-level op)
        @jax.jit
        def run_scan(init_state, draws):
            def step(state, draw):
                return update(state, draw), None

            final_state, _ = jax.lax.scan(step, init_state, draws)
            return final_state

        state = run_scan(init(), draws)
        block = get_moments(state)

        # Verify shape is as expected
        self.assertEqual(np.array(block.mean).shape, (d,))
        self.assertEqual(np.array(block.m2).shape, (d, d))

        # Second jit call — should reuse the compiled function
        key2 = self.next_key()
        draws2 = jnp.asarray(_make_draws(key2, n_steps, d))
        state2 = run_scan(init(), draws2)
        # No exception = shape-static, no recompile triggered by shape change
        block2 = get_moments(state2)
        self.assertEqual(np.array(block2.mean).shape, (d,))


# ---------------------------------------------------------------------------
# 9. f32 merge-accuracy golden (computationally heavy — run LAST)
#
# This test is computationally heavy (d≈400, n≈64k) and is positioned last
# in the test file so that lighter tests run first.
#
# Tolerance derivation (relative):
#   Each Chan merge step introduces relative floating-point error O(ε_mach) into M2.
#   With k merges (k=8), the cumulative relative error is O(k · ε_mach):
#     8 × 1.2e-7 ≈ 1e-6 per entry, relative to |M2[i,j]|.
#   Absolute error per M2 entry ~ 1e-6 × |M2| ~ 1e-6 × n × σ² ≈ 0.06
#   for this test (n=64k, scale~1).  Observed max absolute error ~0.013.
#
#   An absolute tolerance of ≤ 15.0 would be ~1135× too conservative: a
#   structurally broken merge (e.g., dropped cross-terms) can score ~13.95
#   absolute and still pass.  We instead use RELATIVE tolerance (rtol ≈ 1e-4)
#   and anchor downstream: feed f32 vs f64 moments through
#   sample_covariance_eigh_low_rank and assert eigenvalue rtol ≤ 1e-3.
# ---------------------------------------------------------------------------


class F32MergeAccuracyGoldenTest(BlackJAXTest):
    """f32 Chan-merge accuracy vs f64 reference at d≈400, n≈64k."""

    def test_f32_ring_merge_vs_f64_reference(self):
        """f32 Chan-merged M2 vs f64 reference at large (d, n).

        Merges k=8 blocks, each with n_per_block=8000 draws (total n=64000)
        at d=400.

        Pass criteria:
        1. RELATIVE M2 tolerance: max|f32 - f64| / max|f64| ≤ 1e-4.
           This is ~O(k · ε_mach) = ~8×1.2e-7 with ~3 orders of slack.
           Unlike the former absolute ≤15.0 bound, a dropped-cross structural
           bug (scoring ~13.95 absolute, ~2e-4 relative) FAILS this check.
        2. Downstream eigenvalue tolerance: eigenvalues from f32-promoted M2
           vs f64 M2, fed through sample_covariance_eigh_low_rank, agree to
           rtol ≤ 1e-3.
        3. Mean tolerance: max|f32_mean - f64_mean| ≤ 1e-3 (mean is a linear
           statistic, far less sensitive to cancellation than M2).
        """
        k, d, n_per_block = 8, 400, 8000
        keys = jax.random.split(self.next_key(), k)

        draws_list_f64 = [
            np.array(_make_draws(keys[i], n_per_block, d)).astype(np.float64)
            for i in range(k)
        ]
        draws_list_f32 = [d_.astype(np.float32) for d_ in draws_list_f64]

        # f64 reference: single-pass on all draws
        all_draws_f64 = np.concatenate(draws_list_f64, axis=0)
        ref_n, ref_mean_f64, ref_m2_f64 = _ref_single_pass_moments(all_draws_f64)

        # f32 Chan-merge via the production merge_block_ring
        counts_f32, means_f32, m2s_f32 = [], [], []
        for draws in draws_list_f32:
            n, mean, m2 = _ref_single_pass_moments(draws)
            counts_f32.append(np.float32(n))
            means_f32.append(mean.astype(np.float32))
            m2s_f32.append(m2.astype(np.float32))

        counts_arr = jnp.asarray(counts_f32)
        means_arr = jnp.asarray(means_f32)
        m2s_arr = jnp.asarray(m2s_f32)

        merged_f32 = merge_block_ring(counts_arr, means_arr, m2s_arr)

        m2_f32_promoted = np.array(
            merged_f32.m2, dtype=np.float64
        )  # upcast for comparison
        m2_f64 = ref_m2_f64
        n_total = k * n_per_block

        abs_err = np.abs(m2_f32_promoted - m2_f64)
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))
        m2_scale = float(np.max(np.abs(m2_f64)))
        rel_err = max_abs_err / max(m2_scale, 1.0)

        # Report observed error for transparency (visible in verbose test output)
        print(
            f"\nf32 merge-accuracy golden (d={d}, n={n_total}, k={k}):\n"  # noqa: E231
            f"  max |f32 - f64| M2 entry: {format(max_abs_err, '.4g')}\n"
            f"  max relative error: {format(rel_err, '.4g')} (pass criterion: <= 1e-4)\n"
            f"  mean |f32 - f64| M2 entry: {format(mean_abs_err, '.4g')}\n"
            f"  M2 scale (max |f64| entry): {format(m2_scale, '.4g')}"
        )

        # Criterion 1: relative M2 bound (~O(k · eps_mach) with 3 orders of slack)
        self.assertLessEqual(
            rel_err,
            1e-4,
            msg=(
                f"f32 Chan-merge relative error {format(rel_err, '.4g')} > 1e-4 "
                f"(d={d}, n={n_total}, k={k}) -- may indicate dropped cross-terms"
            ),
        )

        # Criterion 2: downstream eigenvalue tolerance under x64.
        # Measured floors on within-dominated data (k=8 blocks, same distribution, 8 seeds):
        #   correct-merge noise ceiling: 4.7e-7
        #   dropped-cross bug floor:     7.3e-5   → 158× separation at threshold 1e-5
        # IMPORTANT: anyone changing this test's data or dimensions must re-derive
        # these floors — the threshold 1e-5 sits 158× below the bug floor only for
        # within-dominated blocks at this (d, k, n_per_block).
        max_rank = 10
        with jax.enable_x64():
            metric_f32 = sample_covariance_eigh_low_rank(
                jnp.asarray(m2_f32_promoted),
                jnp.asarray(float(merged_f32.count)),
                max_rank,
            )
            metric_f64 = sample_covariance_eigh_low_rank(
                jnp.asarray(m2_f64), jnp.asarray(ref_n), max_rank
            )
        np.testing.assert_allclose(
            np.array(metric_f32.lam),
            np.array(metric_f64.lam),
            rtol=1e-5,
            err_msg=(
                f"downstream eigenvalues from f32-merged M2 diverge from f64 reference "
                f"(d={d}, n={n_total}, k={k}, max_rank={max_rank}) -- "
                f"correct-merge ceiling 4.7e-7 < threshold 1e-5 < dropped-cross floor 7.3e-5"
            ),
        )

        # Criterion 3: mean tolerance (linear statistic, far less cancellation)
        mean_f32 = np.array(merged_f32.mean, dtype=np.float64)
        mean_err = float(np.max(np.abs(mean_f32 - ref_mean_f64)))
        self.assertLessEqual(
            mean_err,
            1e-3,
            msg=f"f32 mean error {format(mean_err, '.4g')} > 1e-3",
        )

    def test_f32_ring_merge_between_dominated(self):
        """Between-block-dominated data: eigenvalue criterion fires reliably at rtol=1e-5.

        Same-distribution blocks (the within-dominated case above) have the cross
        term as a ~3e-4 perturbation; Weyl's bound then guarantees eigenvalues shift
        ≤ that fraction → criterion-2 is blind at 1e-5.  Between-dominated blocks
        (distinct block means, ~5σ between-block dispersion) make the cross term
        O(1) of the covariance; dropping it shifts eigenvalues by >> 1e-5.

        This data ensures the eigenvalue criterion is structurally discriminating —
        a dropped-cross bug fails at 34× and the correct merge passes at 158×.
        Anyone changing data/dimension here must re-derive those margins.
        """
        k, d, n_per_block = 8, 20, 4000
        rng = np.random.default_rng(9)
        rho = 0.6
        corr = rho * np.ones((d, d)) + (1 - rho) * np.eye(d)
        L = np.linalg.cholesky(corr)

        # Build k blocks with distinct block means (~5σ between-block dispersion)
        draws_list_f32 = []
        draws_list_f64 = []
        for i in range(k):
            mu = rng.standard_normal(d) * 5.0  # between-block spread >> within std ~1
            z = np.array(jax.random.normal(jax.random.key(100 + i), (n_per_block, d)))
            x_f64 = (z @ L.T + mu).astype(np.float64)
            draws_list_f64.append(x_f64)
            draws_list_f32.append(x_f64.astype(np.float32))

        # f64 reference: single-pass on all draws
        all_f64 = np.concatenate(draws_list_f64, axis=0)
        ref_n, ref_mean_f64, ref_m2_f64 = _ref_single_pass_moments(all_f64)

        # f32 Chan-merged blocks
        counts_f32, means_f32, m2s_f32 = [], [], []
        for draws in draws_list_f32:
            n, mean, m2 = _ref_single_pass_moments(draws)
            counts_f32.append(np.float32(n))
            means_f32.append(mean.astype(np.float32))
            m2s_f32.append(m2.astype(np.float32))

        merged_f32 = merge_block_ring(
            jnp.asarray(counts_f32),
            jnp.asarray(means_f32),
            jnp.asarray(m2s_f32),
        )
        m2_f32_promoted = np.array(merged_f32.m2, dtype=np.float64)
        n_total = k * n_per_block

        # Eigenvalue check: between-dominated cross-term is O(1) of covariance
        # → eigenvalue shift >> 1e-5 for dropped-cross bug (observed: ~7.3e-5 in
        #   within-dominated; between-dominated yields significantly larger margin).
        max_rank = 10
        with jax.enable_x64():
            metric_f32 = sample_covariance_eigh_low_rank(
                jnp.asarray(m2_f32_promoted),
                jnp.asarray(float(merged_f32.count)),
                max_rank,
            )
            metric_f64 = sample_covariance_eigh_low_rank(
                jnp.asarray(ref_m2_f64), jnp.asarray(ref_n), max_rank
            )
        np.testing.assert_allclose(
            np.array(metric_f32.lam),
            np.array(metric_f64.lam),
            rtol=1e-5,
            err_msg=(
                f"between-dominated: eigenvalues diverge from f64 reference "
                f"(d={d}, n={n_total}, k={k}) -- cross-term missing from merge?"
            ),
        )


# ---------------------------------------------------------------------------
# 10. diag_from_moment_block n>=2 guard
# ---------------------------------------------------------------------------


class DiagN2FixTransitionTest(BlackJAXTest):
    """diag_from_moment_block returns ones (not zeros) for n < 2.

    With n=1, M2=0 by definition (single point has no deviation); the old
    n > 0 guard returned zeros — wrong as a step-size proxy (a zero diagonal
    stalls adaptation).  The fix: return ones for n < 2 (safe isotropic
    default).  For n >= 2, behaviour is unchanged.
    """

    def test_n0_returns_ones(self):
        """Empty block (n=0) returns ones (regression guard)."""
        d = 8
        block = MomentBlock(
            count=jnp.zeros(()),
            mean=jnp.zeros((d,)),
            m2=jnp.zeros((d, d)),
        )
        diag = diag_from_moment_block(block)
        np.testing.assert_array_equal(
            np.array(diag),
            np.ones(d),
            err_msg="n=0: diag_from_moment_block must return ones",
        )

    def test_n1_returns_ones(self):
        """Single-sample block (n=1) returns ones — M2=0 is not meaningful."""
        d = 8
        block = MomentBlock(
            count=jnp.ones(()),
            mean=jnp.ones((d,)) * 2.0,
            m2=jnp.zeros((d, d)),  # single point always has M2=0
        )
        diag = diag_from_moment_block(block)
        np.testing.assert_array_equal(
            np.array(diag),
            np.ones(d),
            err_msg="n=1: diag_from_moment_block must return ones (n>=2 required for Bessel)",
        )

    def test_n2_returns_correct_variance(self):
        """n=2 is the first meaningful Bessel-corrected variance; must NOT return ones."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        d = 4
        key = self.next_key()
        draws = _make_draws(key, 2, d).astype(dtype)
        n_f, mean_, m2_ = _ref_single_pass_moments(draws)
        block = _make_block(n_f, mean_, m2_, dtype=dtype)

        diag = diag_from_moment_block(block)
        expected = np.var(draws, axis=0, ddof=1)

        np.testing.assert_allclose(
            np.array(diag),
            expected,
            rtol=1e-4,
            err_msg="n=2: first valid Bessel-corrected variance; should not be ones",
        )
        # Must differ from ones (the distribution has non-unit variance)
        self.assertFalse(
            np.allclose(np.array(diag), np.ones(d)),
            msg="n=2: diag should reflect actual variance, not ones fallback",
        )

    def test_n_large_returns_correct_variance(self):
        """n>=2 behaviour is unchanged from the old n > 0 guard."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        d, n = 15, 100
        key = self.next_key()
        draws = _make_draws(key, n, d).astype(dtype)
        n_f, mean_, m2_ = _ref_single_pass_moments(draws)
        block = _make_block(n_f, mean_, m2_, dtype=dtype)

        diag = diag_from_moment_block(block)
        expected = np.var(draws, axis=0, ddof=1)

        np.testing.assert_allclose(
            np.array(diag),
            expected,
            rtol=1e-4,
            err_msg="n>=2: diag_from_moment_block must equal Bessel-corrected variance",
        )

    def test_n1_via_reset_window_get_diag_reference(self):
        """After a single update, get_diag_reference returns ones (not zeros).

        This is the concrete step-size proxy scenario: the first update call
        sets n=1; the proxy must not return zeros (which would stall adaptation).
        """
        d = 8
        key = self.next_key()
        single_draw = np.array(jax.random.normal(key, (d,)))

        init, update, _, _, _, get_diag_ref = reset_window_buffer(d)
        state = init()
        state = update(state, jnp.asarray(single_draw))
        diag = get_diag_ref(state)

        np.testing.assert_array_equal(
            np.array(diag),
            np.ones(d),
            err_msg="Single-sample: diag_ref must be ones (not zeros) for safe step-size proxy",
        )


# ---------------------------------------------------------------------------
# 11. merge_block_ring k=1 short-circuit
# ---------------------------------------------------------------------------


class MergeBlockRingK1ShortCircuitTest(BlackJAXTest):
    """merge_block_ring at k=1 takes a static Python short-circuit (not a scan).

    The short-circuit returns MomentBlock(counts[0], means[0], m2s[0]) directly,
    which is bit-identical to what a length-1 scan would produce but avoids
    ~1.6× compile overhead at d=400 for the reset_window_buffer use-case.
    """

    def test_k1_returns_single_slot_contents(self):
        """merge_block_ring at k=1 returns the single slot, unmodified."""
        d, n = 20, 50
        key = self.next_key()
        draws = jax.random.normal(key, (n, d))
        mean = jnp.mean(draws, axis=0)
        m2 = (draws - mean).T @ (draws - mean)

        counts = jnp.array([float(n)])
        means = jnp.array([np.array(mean)])
        m2s = jnp.array([np.array(m2)])

        merged = merge_block_ring(counts, means, m2s)

        np.testing.assert_array_equal(
            np.array(merged.count),
            np.array(counts[0]),
            err_msg="k=1: count must equal the single slot count",
        )
        np.testing.assert_array_equal(
            np.array(merged.mean),
            np.array(means[0]),
            err_msg="k=1: mean must equal the single slot mean (bit-identical)",
        )
        np.testing.assert_array_equal(
            np.array(merged.m2),
            np.array(m2s[0]),
            err_msg="k=1: m2 must equal the single slot m2 (bit-identical)",
        )

    def test_k1_shape_dtype_contract(self):
        """Short-circuit produces correct shapes and dtypes on empty slot."""
        d = 30
        counts = jnp.zeros((1,))
        means = jnp.zeros((1, d))
        m2s = jnp.zeros((1, d, d))

        merged = merge_block_ring(counts, means, m2s)

        self.assertEqual(merged.count.shape, ())
        self.assertEqual(merged.mean.shape, (d,))
        self.assertEqual(merged.m2.shape, (d, d))
        self.assertEqual(merged.count.dtype, counts.dtype)
        self.assertEqual(merged.mean.dtype, means.dtype)
        self.assertEqual(merged.m2.dtype, m2s.dtype)

    def test_split_pop_k1_degenerate(self):
        """accumulating_split_pop_buffer at k=1 is identical to reset_window_buffer.

        push_split at k=1 advances write_pos: (0+1)%1 = 0, then zeroes slot 0.
        Net effect: the single slot is zeroed — same as reset_window hard reset.
        """
        d, n = 10, 30
        key1, key2 = jax.random.split(self.next_key(), 2)
        draws1 = _make_draws(key1, n, d)
        draws2 = _make_draws(key2, n, d)

        init, update, push_split, get_moments, _, _ = accumulating_split_pop_buffer(
            d, k=1
        )
        state = init()

        # Window 1
        for i in range(n):
            state = update(state, jnp.asarray(draws1[i]))
        state = push_split(state)  # hard reset (k=1)

        # Window 2: should see ONLY draws2, not draws1
        for i in range(n):
            state = update(state, jnp.asarray(draws2[i]))
        block = get_moments(state)

        ref_n, ref_mean, ref_m2 = _ref_single_pass_moments(draws2)

        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="k=1 split_pop: push_split must zero the slot (hard reset)",
        )
        self.assertAlmostEqual(
            float(block.count),
            ref_n,
            places=5,
            msg="k=1 split_pop: count must reflect draws2 only (draws1 zeroed on push_split)",
        )


# ---------------------------------------------------------------------------
# 12. late_start multi-window interaction
# ---------------------------------------------------------------------------


class LateStartMultiWindowTest(BlackJAXTest):
    """late_start × reset_window across ≥ 2 windows.

    push_split resets the skip counter so that each new window has its own
    independent offset period — the late-start skip is NOT cumulative across
    windows.
    """

    def test_two_windows_each_get_independent_late_start(self):
        """After push_split, window 2 has a fresh offset period.

        Window 1: offset=5, n_draws=20 → accumulates draws[5:20]
        Window 2: push_split → inner reset + skip counter → 0
                  offset=5, n_draws=20 → accumulates draws2[5:20] only
        """
        d, offset, n = 8, 5, 20
        key1, key2 = jax.random.split(self.next_key(), 2)
        draws1 = _make_draws(key1, n, d)
        draws2 = _make_draws(key2, n, d)

        inner_fns = reset_window_buffer(d)
        ls_init, ls_update, ls_push, ls_get_moments, _, _ = late_start(
            inner_fns, offset_steps=offset
        )
        state = ls_init()

        for i in range(n):
            state = ls_update(state, jnp.asarray(draws1[i]))
        block_w1 = ls_get_moments(state)

        ref1_n, ref1_mean, _ = _ref_single_pass_moments(draws1[offset:])
        np.testing.assert_allclose(
            np.array(block_w1.mean),
            ref1_mean,
            rtol=1e-4,
            err_msg="window 1: mean should be draws1[offset:]",
        )
        self.assertAlmostEqual(float(block_w1.count), ref1_n, places=5)

        state = ls_push(state)  # resets inner accumulator AND skip counter

        for i in range(n):
            state = ls_update(state, jnp.asarray(draws2[i]))
        block_w2 = ls_get_moments(state)

        ref2_n, ref2_mean, _ = _ref_single_pass_moments(draws2[offset:])
        np.testing.assert_allclose(
            np.array(block_w2.mean),
            ref2_mean,
            rtol=1e-4,
            err_msg="window 2: must NOT be contaminated by window 1",
        )
        self.assertAlmostEqual(
            float(block_w2.count),
            ref2_n,
            places=5,
            msg="window 2: count = n-offset (fresh independent offset)",
        )

    def test_num_skipped_resets_on_push_split(self):
        """push_split resets num_skipped to 0 so the new window gets a full offset."""
        d, offset = 5, 10
        inner_fns = reset_window_buffer(d)
        ls_init, ls_update, ls_push, _, _, _ = late_start(
            inner_fns, offset_steps=offset
        )
        state = ls_init()

        key = self.next_key()
        draws = jax.random.normal(key, (offset + 5, d))
        for i in range(offset + 5):
            state = ls_update(state, draws[i])

        self.assertEqual(int(state.num_skipped), offset)

        state = ls_push(state)
        self.assertEqual(
            int(state.num_skipped),
            0,
            msg="push_split must reset num_skipped so next window gets full offset",
        )


# ---------------------------------------------------------------------------
# 13. Read-before-push ordering: violation behavior
# ---------------------------------------------------------------------------


class ReadBeforePushOrderingTest(BlackJAXTest):
    """Documents the observed behavior when the read-before-push contract is violated.

    The module contract: callers MUST call get_moments BEFORE push_split.
    These tests document what actually happens on violation as durable failure-mode
    goldens, not as endorsements of the pattern.
    """

    def test_reset_window_push_before_read_returns_empty_block(self):
        """reset_window: push_split BEFORE get_moments → empty block (all data lost).

        reset_window's push_split zeroes the single accumulator immediately;
        get_moments after push_split sees count=0 everywhere.
        """
        d, n = 8, 30
        key = self.next_key()
        init, update, push_split, get_moments, _, _ = reset_window_buffer(d)
        state = init()
        draws = jax.random.normal(key, (n, d))
        for i in range(n):
            state = update(state, draws[i])

        state_after_push = push_split(state)  # VIOLATION: push before read
        block_after_push = get_moments(state_after_push)

        self.assertAlmostEqual(
            float(block_after_push.count),
            0.0,
            places=8,
            msg="reset_window push-before-read: count=0 (all data lost)",
        )
        np.testing.assert_array_equal(
            np.array(block_after_push.mean),
            np.zeros((d,)),
            err_msg="reset_window push-before-read: mean zeroed (all data lost)",
        )

    def test_split_pop_push_before_read_loses_oldest_split(self):
        """split_pop: push_split BEFORE get_moments → oldest split overwritten.

        When the ring is full (k splits accumulated), push_split advances
        write_pos to (old_wp+1)%k and zeroes that slot — the oldest completed
        split.  Calling it before read loses one split silently.

        Unlike reset_window (catastrophic: entire buffer zeroed), here k-1 splits
        remain.  The merged block after violation equals the k-1 retained splits.
        """
        d, k, n_per_split = 6, 3, 20
        keys = jax.random.split(self.next_key(), k)
        draws_list = [
            np.array(jax.random.normal(keys[i], (n_per_split, d))) for i in range(k)
        ]

        init, update, push_split, get_moments, _, _ = accumulating_split_pop_buffer(
            d, k
        )
        state = init()

        for split_idx in range(k - 1):
            for i in range(n_per_split):
                state = update(state, jnp.asarray(draws_list[split_idx][i]))
            state = push_split(state)
        for i in range(n_per_split):
            state = update(state, jnp.asarray(draws_list[k - 1][i]))

        # Pre-violation: all k splits present
        block_correct = get_moments(state)
        self.assertAlmostEqual(
            float(block_correct.count), float(k * n_per_split), places=5
        )

        # VIOLATION: push before read
        state_after_push = push_split(state)
        block_after_push = get_moments(state_after_push)

        # Oldest split (draws_list[0]) is gone; k-1 remain
        retained = np.concatenate(draws_list[1:], axis=0)
        ref_n, ref_mean, _ = _ref_single_pass_moments(retained)

        self.assertAlmostEqual(
            float(block_after_push.count),
            float((k - 1) * n_per_split),
            places=5,
            msg="split_pop push-before-read: oldest split lost, k-1 remain",
        )
        np.testing.assert_allclose(
            np.array(block_after_push.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="split_pop push-before-read: merged = k-1 retained splits",
        )


# ---------------------------------------------------------------------------
# 14. late_start × ensemble_batch offset semantics
# ---------------------------------------------------------------------------


class LateStartEnsembleOffsetSemanticsTest(BlackJAXTest):
    """late_start × ensemble_batch_buffer: offset_steps counts update CALLS, not draws.

    Each update call feeds (n_chains, d) — n_chains draws at once.
    offset_steps counts how many calls are skipped, not how many total draws.
    """

    def test_offset_counts_calls_not_draws(self):
        """offset_steps=5 skips 5 calls (5 × n_chains draws), not 5 draws."""
        d, n_chains, k = 8, 32, 3
        offset, n_calls = 5, 20
        key = self.next_key()
        keys = jax.random.split(key, n_calls)
        batches = [
            np.array(jax.random.normal(keys[i], (n_chains, d))) for i in range(n_calls)
        ]

        inner = ensemble_batch_buffer(d, n_chains, k)
        ls_init, ls_update, _, ls_get_moments, _, _ = late_start(
            inner, offset_steps=offset
        )
        state = ls_init()

        for i in range(n_calls):
            state = ls_update(state, jnp.asarray(batches[i]))

        block = ls_get_moments(state)

        # offset calls × n_chains chains = offset*n_chains draws skipped
        kept = np.concatenate(batches[offset:], axis=0)
        ref_n, ref_mean, _ = _ref_single_pass_moments(kept)

        self.assertAlmostEqual(
            float(block.count),
            ref_n,
            places=4,
            msg=(
                f"offset_steps={offset} skips {offset} calls "
                f"(= {offset * n_chains} draws, NOT {offset} draws)"
            ),
        )
        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="late_start × ensemble_batch: mean wrong (offset counts calls)",
        )

    def test_meads_late_window_semantics(self):
        """MEADS late-window fraction: offset_steps = window_size // 2 (step-level)."""
        d, n_chains = 10, 16
        window_size, k = 20, 3
        offset_steps = window_size // 2  # 10 step-calls = 10 × 16 draws skipped
        key = self.next_key()
        keys = jax.random.split(key, window_size)
        batches = [
            np.array(jax.random.normal(keys[i], (n_chains, d)))
            for i in range(window_size)
        ]

        inner = ensemble_batch_buffer(d, n_chains, k)
        ls_init, ls_update, _, ls_get_moments, _, _ = late_start(
            inner, offset_steps=offset_steps
        )
        state = ls_init()
        for batch in batches:
            state = ls_update(state, jnp.asarray(batch))

        block = ls_get_moments(state)

        kept = np.concatenate(batches[offset_steps:], axis=0)
        ref_n, ref_mean, _ = _ref_single_pass_moments(kept)

        self.assertAlmostEqual(float(block.count), ref_n, places=4)
        np.testing.assert_allclose(
            np.array(block.mean),
            ref_mean,
            rtol=1e-4,
            err_msg="MEADS late-window: mean wrong",
        )


# ---------------------------------------------------------------------------
# 15. ensemble_batch_buffer shape guard
# ---------------------------------------------------------------------------


class EnsembleBatchShapeGuardTest(BlackJAXTest):
    """ensemble_batch_buffer: wrong batch shape raises ValueError at trace time."""

    def test_wrong_batch_shape_raises_valueerror(self):
        """Batch with wrong n_chains raises ValueError at first update (trace time)."""
        d, n_chains, k = 8, 32, 3
        key = self.next_key()
        half_batch = jnp.asarray(np.array(jax.random.normal(key, (n_chains // 2, d))))

        init, update, _, _, _, _ = ensemble_batch_buffer(d, n_chains, k)
        state = init()

        with self.assertRaises(ValueError):
            update(state, half_batch)  # batch.shape[0] = 16 != n_chains = 32

    def test_correct_batch_shape_passes(self):
        """Correct (n_chains, d) batch is accepted and produces count=n_chains."""
        d, n_chains, k = 8, 32, 3
        key = self.next_key()
        batch = jnp.asarray(np.array(jax.random.normal(key, (n_chains, d))))

        init, update, _, _, get_support, _ = ensemble_batch_buffer(d, n_chains, k)
        state = init()
        state = update(state, batch)
        total, _ = get_support(state)
        self.assertAlmostEqual(float(total), float(n_chains), places=5)


# ---------------------------------------------------------------------------
# 16. MEADS end-to-end parity: old accumulator vs ensemble_batch_buffer
# ---------------------------------------------------------------------------

# Frozen copy of the bespoke Chan/Welford accumulator that lived in
# meads_adaptation.py before R3-final.  Embedded here (not imported) so that
# any future mutation to the production module cannot alias the golden.


class _FrozenLRDAccumulatorState(NamedTuple):
    """Pre-R3-final Chan/Welford state: (mean, m2, count)."""

    mean: jnp.ndarray
    m2: jnp.ndarray
    count: jnp.ndarray


def _frozen_lrd_accumulator_init(d: int) -> _FrozenLRDAccumulatorState:
    return _FrozenLRDAccumulatorState(
        mean=jnp.zeros((d,)), m2=jnp.zeros((d, d)), count=jnp.zeros(())
    )


def _frozen_lrd_accumulator_update(
    acc: _FrozenLRDAccumulatorState, batch: jnp.ndarray
) -> _FrozenLRDAccumulatorState:
    mean_a, m2_a, n_a = acc
    n_b = batch.shape[0]
    mean_b = jnp.mean(batch, axis=0)
    centered_b = batch - mean_b[None, :]
    m2_b = centered_b.T @ centered_b
    delta = mean_b - mean_a
    n_ab = n_a + n_b
    mean_ab = mean_a + delta * (n_b / n_ab)
    m2_ab = m2_a + m2_b + jnp.outer(delta, delta) * (n_a * n_b / n_ab)
    return _FrozenLRDAccumulatorState(mean=mean_ab, m2=m2_ab, count=n_ab)


class MeadsBufferEndToEndParityTest(BlackJAXTest):
    """Old Chan/Welford accumulator matches ensemble_batch_buffer at the LRD output level.

    Path A: frozen pre-R3-final accumulator → sample_covariance_eigh_low_rank
    Path B: ensemble_batch_buffer → get_moments → sample_covariance_eigh_low_rank

    Both paths must produce bit-identical (sigma, U, lam) under x64.
    The frozen inline copy prevents silent aliasing: if the production code
    changes, path A still reflects the original math.
    """

    def test_end_to_end_lrd_parity_x64(self):
        """ensemble_batch_buffer matches old Chan/Welford at LRD output under x64."""
        d, n_chains, k = 12, 16, 1
        n_steps = 30
        window_start = n_steps // 2  # MEADS-like: skip first half

        key = self.next_key()
        keys = jax.random.split(key, n_steps)
        batches = [
            np.array(jax.random.normal(keys[i], (n_chains, d))) for i in range(n_steps)
        ]

        with jax.enable_x64():
            # Path A: frozen inline Chan/Welford (the pre-R3-final math)
            acc = _frozen_lrd_accumulator_init(d)
            for i in range(window_start, n_steps):
                acc = _frozen_lrd_accumulator_update(acc, jnp.asarray(batches[i]))
            metric_a = sample_covariance_eigh_low_rank(acc.m2, acc.count, k)

            # Path B: ensemble_batch_buffer (the new D-layer path)
            init, update, _, get_moments, _, _ = ensemble_batch_buffer(d, n_chains, k)
            state = init()
            for i in range(window_start, n_steps):
                state = update(state, jnp.asarray(batches[i]))
            block = get_moments(state)
            metric_b = sample_covariance_eigh_low_rank(block.m2, block.count, k)

        np.testing.assert_array_equal(
            np.array(metric_a.sigma),
            np.array(metric_b.sigma),
            err_msg="MEADS parity: sigma differs (old accumulator vs ensemble_batch_buffer)",
        )
        np.testing.assert_array_equal(
            np.array(metric_a.U),
            np.array(metric_b.U),
            err_msg="MEADS parity: U differs (old accumulator vs ensemble_batch_buffer)",
        )
        np.testing.assert_array_equal(
            np.array(metric_a.lam),
            np.array(metric_b.lam),
            err_msg="MEADS parity: lam differs (old accumulator vs ensemble_batch_buffer)",
        )


if __name__ == "__main__":
    absltest.main()
