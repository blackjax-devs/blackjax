from typing import Dict, List, Tuple, Union

import jax.numpy as jnp

Array = Union[jnp.ndarray, jnp.DeviceArray, float]
PyTree = Union[Dict, List, Tuple, Array]
