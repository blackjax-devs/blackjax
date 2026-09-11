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
"""Auto-provision >=2 JAX devices for the multi-device tests.

These tests require at least 2 devices. The right way to obtain them depends on
the machine, and the choice must be made *before* JAX is imported — so it lives
here in ``conftest.py``, whose module-level code runs at collection time, ahead
of the test modules' ``import jax``.

Precedence:
  1. An explicit user-set ``XLA_FLAGS`` is respected (never overridden).
  2. If >=2 real GPUs are visible, do nothing — the tests run on the real
     multi-GPU mesh (``jax.devices()`` already returns them).
  3. Otherwise, simulate 2 CPU devices via
     ``--xla_force_host_platform_device_count=2`` so the tests still run on a
     single-device machine (typical CI / laptop).

Effect: ``pytest tests/test_multidevice/`` does the right thing everywhere with
no flag to remember — real GPUs where present, simulated CPU otherwise — and the
CI workflow no longer needs to set ``XLA_FLAGS`` explicitly.

Limitation: a machine with >=2 physical GPUs but a CPU-only ``jaxlib`` install is
not auto-handled (nvidia-smi reports the GPUs, yet JAX exposes 1 CPU device, so
the tests skip). Install a CUDA ``jaxlib`` or set ``XLA_FLAGS`` explicitly.
"""
import os
import subprocess


def _has_multi_gpu() -> bool:
    """Return True if >=2 GPUs are visible, without importing JAX.

    Importing JAX to count devices would lock in the device set before we could
    request host-platform simulation, so we probe ``nvidia-smi`` instead.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return sum(line.startswith("GPU ") for line in result.stdout.splitlines()) >= 2


if "XLA_FLAGS" not in os.environ and not _has_multi_gpu():
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
