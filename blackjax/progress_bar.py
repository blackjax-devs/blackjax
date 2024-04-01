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
import tqdm
from tqdm.auto import tqdm as tqdm_auto
import jax
from jax import lax


def progress_bar_scan(num_samples, num_chains=1, print_rate=None):
    """Factory that builds a progress bar decorator along
    with the `set_tqdm_description` and `close_tqdm` functions
    """

    if print_rate is None:
        if num_samples > 20:
            print_rate = int(num_samples / 20)
        else:
            print_rate = 1  # if you run the sampler for less than 20 iterations

    remainder = num_samples % print_rate

    tqdm_bars = {}
    for chain in range(num_chains):
        tqdm_bars[chain] = tqdm_auto(range(num_samples), position=chain)
        tqdm_bars[chain].set_description("Compiling.. ", refresh=True)

    def _update_tqdm(arg, chain):
        chain = int(chain)
        tqdm_bars[chain].set_description(f"Running chain {chain}", refresh=False)
        tqdm_bars[chain].update(arg)

    def _close_tqdm(arg, chain):
        chain = int(chain)
        tqdm_bars[chain].update(arg)
        tqdm_bars[chain].close()

    def _update_progress_bar(iter_num, chain):
        """Updates tqdm progress bar of a JAX loop only if the iteration number is a multiple of the print_rate
        Usage: carry = progress_bar((iter_num, print_rate), carry)
        """

        _ = lax.cond(
            iter_num == 0,
            lambda _: jax.debug.callback(_update_tqdm, iter_num, chain),
            lambda _: None,
            operand=None,
        )
        _ = lax.cond(
            (iter_num % print_rate) == 0,
            lambda _: jax.debug.callback(_update_tqdm, print_rate, chain),
            lambda _: None,
            operand=None,
        )
        _ = lax.cond(
            iter_num == num_samples - 1,
            lambda _: jax.debug.callback(_close_tqdm, remainder, chain),
            lambda _: None,
            operand=None,
        )

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        looping over a tuple whose elements are `np.arange(num_samples), and a
        chain id defined as `chain * np.ones(num_samples)`, or be looping over a
        tuple who's first element and second elements include iter_num and chain.
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                if num_chains > 1:
                    iter_num, chain, *_ = x
                else:
                    iter_num, *_ = x
                    chain = 0
            else:
                iter_num = x
                chain = 0
            _update_progress_bar(iter_num, chain)
            return func(carry, x)

        return wrapper_progress_bar

    return _progress_bar_scan
