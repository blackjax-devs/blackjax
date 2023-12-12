# Copyright 2020- The Blackjax Authors.
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
from typing import Any, Iterable, Mapping, Union

import jax
from jax.typing import ArrayLike

"""
Following the current best practice (https://jax.readthedocs.io/en/latest/jax.typing.html)
We use:
- `ArrayLike` and `ArrayLikeTree` to annotate function input,
- `Array` and `ArrayTree` to annotate function output.

Leaves of a Pytree definition in the library are in principle annotated as
`Array`, as they are mostly internal representation. For example:
```
class WelfordAlgorithmState(NamedTuple):
    mean: Array
    ...
```

[TODO] Improve scalar-like typing (e.g. `logdensity`, `acceptance_rate`).
While they are `Array` (as in most cases they should be output of a Jax
function), we annotate them as `float` to empathizes they should be scalar
(until we introduce shape annotation).
"""
#: JAX PyTrees
Array = jax.Array
ArrayTree = Union[jax.Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayLikeTree = Union[
    ArrayLike, Iterable["ArrayLikeTree"], Mapping[Any, "ArrayLikeTree"]
]

#: JAX PRNGKey
PRNGKey = jax.Array
