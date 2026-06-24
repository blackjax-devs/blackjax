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
