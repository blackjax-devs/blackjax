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
"""Progress bar decorators for use with step functions.
Adapted from Jeremie Coullon's blog post :cite:p:`progress_bar`.
"""
from threading import Lock

from fastprogress.fastprogress import progress_bar
from jax import lax
from jax.experimental import io_callback
from jax.numpy import array


def progress_bar_scan(num_samples, print_rate=None):
    "Progress bar for a JAX scan"
    progress_bars = {}
    idx_counter = 0
    lock = Lock()

    if print_rate is None:
        if num_samples > 20:
            print_rate = int(num_samples / 20)
        else:
            print_rate = 1  # if you run the sampler for less than 20 iterations

    def _calc_chain_idx(iter_num):
        nonlocal idx_counter
        with lock:
            idx = idx_counter
            idx_counter += 1
        return idx

    def _update_bar(arg, chain_id):
        chain_id = int(chain_id)
        if arg == 0:
            chain_id = _calc_chain_idx(arg)
            progress_bars[chain_id] = progress_bar(range(num_samples))
            progress_bars[chain_id].update(0)

        progress_bars[chain_id].update_bar(arg + 1)
        return chain_id

    def _close_bar(arg, chain_id):
        progress_bars[int(chain_id)].on_iter_end()

    def _update_progress_bar(iter_num, chain_id):
        "Updates progress bar of a JAX scan or loop"

        chain_id = lax.cond(
            # update every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) | (iter_num == (num_samples - 1)),
            lambda _: io_callback(_update_bar, array(0), iter_num, chain_id),
            lambda _: chain_id,
            operand=None,
        )

        _ = lax.cond(
            iter_num == num_samples - 1,
            lambda _: io_callback(_close_bar, None, iter_num + 1, chain_id),
            lambda _: None,
            operand=None,
        )
        return chain_id

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            subcarry, chain_id = carry
            chain_id = _update_progress_bar(iter_num, chain_id)
            subcarry, y = func(subcarry, x)

            return (subcarry, chain_id), y

        return wrapper_progress_bar

    return _progress_bar_scan


def gen_scan_fn(num_samples, progress_bar, print_rate=None):
    if progress_bar:

        def scan_wrap(f, init, *args, **kwargs):
            func = progress_bar_scan(num_samples, print_rate)(f)
            carry = (init, -1)
            (last_state, _), output = lax.scan(func, carry, *args, **kwargs)
            return last_state, output

        return scan_wrap
    else:
        return lax.scan
