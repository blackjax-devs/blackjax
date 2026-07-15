# Copyright 2024- The Blackjax Authors.
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
"""Shared pytest fixtures for blackjax/tests/adaptation/."""
import jax
import pytest


@pytest.fixture(autouse=True)
def restore_jax_config():
    """Restore JAX config state after each test (hermetic to x64 leakage).

    Some tests enable x64 via jax.enable_x64() context manager for precision-
    sensitive tests. While context managers should properly restore state, this
    fixture provides defense-in-depth: if a test exits abnormally (exception,
    timeout) or a context manager has an edge-case bug, this ensures x64 state
    is restored for the next test.  This prevents flaky failures when tests run
    under different JAX configs depending on execution order.
    """
    orig_x64 = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", orig_x64)


@pytest.fixture(autouse=True)
def clear_jax_caches():
    """Clear JAX JIT/XLA caches between adaptation tests.

    Adaptation tests compile many structurally-different traced functions
    (different num_chains, n_paths, dispatch paths).  Without cache clearing
    the xdist worker that runs these tests accumulates JIT traces across tests,
    which can OOM the GitHub-hosted runner (7 GB) when multiple workers run
    heavy adaptation tests in parallel.

    Clearing after each test keeps per-worker peak RSS under the CI limit.
    Note: this makes sequential single-process runs slower (each test re-JITs),
    but CI uses xdist workers where each test already starts with a cold cache.
    """
    yield
    jax.clear_caches()
