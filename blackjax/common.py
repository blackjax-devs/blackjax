from typing import Union, Tuple, List, Dict
import jax.numpy as jnp

Array = Union[jnp.ndarray, jnp.DeviceArray, float]
PyTree = Union[Dict, List, Tuple, Array]
