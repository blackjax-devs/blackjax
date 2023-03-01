from typing import Any, Iterable, Mapping, Union

from chex import Array
import jax

#: JAX PyTrees
PyTree = Union[Array, Iterable["PyTree"], Mapping[Any, "PyTree"]]

#: JAX PRNGKey
PRNGKey = jax.random.PRNGKeyArray
