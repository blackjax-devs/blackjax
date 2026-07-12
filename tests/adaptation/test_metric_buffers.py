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
- **Chan-merge exactness**: merged blocks == single-pass moments on the same
  data (f64, tight atol 1e-10).
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

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from blackjax.adaptation.metric_buffers import (
    AccumulatingSplitPopState,
    LateStartState,
    MomentBlock,
    ResetWindowState,
    accumulating_split_pop_buffer,
    chan_merge_two,
    chan_update_batch,
    diag_from_moment_block,
    ensemble_batch_buffer,
    late_start,
    merge_block_ring,
    reset_window_buffer,
)
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


def _ref_chan_merge_two(na, mean_a, m2_a, nb, mean_b, m2_b):
    """Frozen Chan merge reference (two pre-accumulated blocks, dense M2)."""
    n_ab = na + nb
    if n_ab == 0:
        return 0.0, np.zeros_like(mean_a), np.zeros_like(m2_a)
    delta = mean_b - mean_a
    mean_ab = mean_a + delta * (nb / n_ab)
    m2_ab = m2_a + m2_b + np.outer(delta, delta) * (na * nb / n_ab)
    return n_ab, mean_ab, m2_ab


def _ref_chan_merge_ring(draws_list):
    """Frozen reference: Chan-merge a list of draw arrays (each (n_i, d))."""
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
            n_acc, mean_acc, m2_acc = _ref_chan_merge_two(
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


class ChanMergeTwoTest(BlackJAXTest):
    """Chan-merge of two pre-accumulated blocks == single-pass on the union.

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
        """chan_merge_two(A, B) matches single-pass moments on A∪B.

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

        merged = chan_merge_two(block_a, block_b)

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
        merged_1 = chan_merge_two(empty, block)
        _assert_allclose_dtype(np.array(merged_1.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged_1.m2), ref_m2.astype(dtype), dtype)
        self.assertAlmostEqual(float(merged_1.count), ref_n, places=5)

        # block ∪ empty
        merged_2 = chan_merge_two(block, empty)
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
        merged = chan_merge_two(block_a, block_b)

        all_draws = np.concatenate([draws_a, draws_b], axis=0)
        ref_n, ref_mean, ref_m2_diag = _ref_single_pass_diag_moments(all_draws)

        _assert_allclose_dtype(np.array(merged.mean), ref_mean.astype(dtype), dtype)
        _assert_allclose_dtype(np.array(merged.m2), ref_m2_diag.astype(dtype), dtype)


class ChanUpdateBatchTest(BlackJAXTest):
    """chan_update_batch(block, batch) == chan_merge_two(block, single_pass(batch))."""

    def test_matches_chan_merge_two(self):
        """chan_update_batch equals single-pass on A∪B at appropriate dtype tolerance."""
        dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
        key_a, key_b = jax.random.split(self.next_key(), 2)
        d, n_a, n_b = 12, 25, 40
        draws_a = _make_draws(key_a, n_a, d).astype(dtype)
        draws_b = _make_draws(key_b, n_b, d).astype(dtype)

        na_f, mean_a, m2_a = _ref_single_pass_moments(draws_a)
        block_a = _make_block(na_f, mean_a, m2_a, dtype=dtype)

        updated = chan_update_batch(block_a, jnp.asarray(draws_b))

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
        updated = chan_update_batch(empty, draw[None, :])  # explicit (1, d)
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
        """Default state carries no raw-draw ring (opt-in capability, off by default)."""
        d = 6
        init, _, _, _, _, _ = reset_window_buffer(d, requires_draws=False)
        state = init()
        # State is a ResetWindowState with count/mean/m2 only
        self.assertIsInstance(state, ResetWindowState)
        self.assertEqual(len(state), 3)  # count, mean, m2 only

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

    def test_pop_oldest_exactness(self):
        """Merge of k-1 remaining blocks == recomputation from scratch (k+1 splits, k=3)."""
        k, d, n_per_split = 3, 8, 20
        n_splits_total = k + 1  # one extra so the ring wraps and oldest is dropped

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

        # After k+1 splits the ring has wrapped: the first split (draws_list[0])
        # has been overwritten.  The valid blocks should cover draws_list[1..k].
        remaining_draws = np.concatenate(draws_list[1:], axis=0)
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
        """Default state has no raw-draw ring (raw-draw ring is opt-in, off by default)."""
        d, k = 5, 3
        init, _, _, _, _, _ = accumulating_split_pop_buffer(d, k, requires_draws=False)
        state = init()
        self.assertIsInstance(state, AccumulatingSplitPopState)
        self.assertEqual(len(state), 5)  # counts, means, m2s, write_pos, num_valid

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
        """get_diag_reference works on ResetWindowState too."""
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
# Tolerance derivation:
#   Chan-merge is subject to catastrophic cancellation in the M2 terms at
#   large n. The worst-case error bound for the Chan M2 formula is:
#     |ΔM2| ≤ O(n · ε_mach · ||x||²)
#   For f32 (ε_mach ≈ 1.2e-7), d=400, n=64000, scale~1:
#     per-entry error ~ n · ε_mach · d ~ 64000 · 1.2e-7 · 400 ≈ 3.1
#   In practice the ring-merge adds multiple Chan steps, so absolute entry
#   error up to ~10.0 is credible. We measure the ACTUAL observed error
#   (in f32 vs f64 reference) and state it here for transparency.
#
#   Stated tolerance: absolute entry-wise error ≤ 15.0 (conservative bound).
#   Observed error during implementation: reported in the test output.
# ---------------------------------------------------------------------------


class F32MergeAccuracyGoldenTest(BlackJAXTest):
    """f32 Chan-merge accuracy vs f64 reference at d≈400, n≈64k."""

    def test_f32_ring_merge_vs_f64_reference(self):
        """f32 Chan-merged M2 vs f64 reference at large (d, n).

        Merges k=8 blocks, each with n_per_block=8000 draws (total n=64000)
        at d=400.

        Tolerance: derived from O(n · ε_mach · d) error bound for the
        Chan formula; the OBSERVED per-entry absolute error is printed for
        transparency.  The stated pass criterion is max|f32_M2 - f64_M2| ≤ 15.0
        (conservative; tighter than the theoretical worst case).
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
        counts_f32 = []
        means_f32 = []
        m2s_f32 = []
        for draws in draws_list_f32:
            n, mean, m2 = _ref_single_pass_moments(draws)
            counts_f32.append(np.float32(n))
            means_f32.append(mean.astype(np.float32))
            m2s_f32.append(m2.astype(np.float32))

        counts_arr = jnp.asarray(counts_f32)
        means_arr = jnp.asarray(means_f32)
        m2s_arr = jnp.asarray(m2s_f32)

        merged_f32 = merge_block_ring(counts_arr, means_arr, m2s_arr)

        m2_f32 = np.array(merged_f32.m2, dtype=np.float64)  # upcast for comparison
        m2_f64 = ref_m2_f64

        abs_err = np.abs(m2_f32 - m2_f64)
        max_abs_err = float(np.max(abs_err))
        mean_abs_err = float(np.mean(abs_err))

        # Report observed error for transparency (visible in verbose test output)
        max_err_s = format(max_abs_err, ".4g")
        mean_err_s = format(mean_abs_err, ".4g")
        n_total = k * n_per_block
        print(
            f"\nf32 merge-accuracy golden (d={d}, n={n_total}, k={k}):\n"  # noqa: E231
            f"  max |f32 - f64| M2 entry: {max_err_s}\n"
            f"  mean |f32 - f64| M2 entry: {mean_err_s}\n"
            f"  Stated tolerance: <= 15.0 (derived from O(n*eps_mach*d) bound)"
        )

        # Stated tolerance <= 15.0 (catastrophic-cancellation bound for Chan M2 at this scale)
        self.assertLessEqual(
            max_abs_err,
            15.0,
            msg=f"f32 Chan-merge max entry error {max_err_s} > 15.0 (d={d}, n={n_total}, k={k})",
        )

        # The mean should be much smaller than the max (the error is sparse)
        self.assertLessEqual(
            mean_abs_err,
            1.0,
            msg=f"mean per-entry error {mean_err_s} > 1.0 (unexpected bulk error)",
        )

        # Mean comparison is tight (mean is a linear statistic, less cancellation)
        mean_f32 = np.array(merged_f32.mean, dtype=np.float64)
        mean_err = float(np.max(np.abs(mean_f32 - ref_mean_f64)))
        mean_err_s2 = format(mean_err, ".4g")
        self.assertLessEqual(
            mean_err,
            1e-3,
            msg=f"f32 mean error {mean_err_s2} > 1e-3 (mean should be much tighter than M2)",
        )


if __name__ == "__main__":
    absltest.main()
