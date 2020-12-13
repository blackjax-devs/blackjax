# BlackJAX

# Build a NUTS sampler in 3 lines of code

```python
from blackjax import nuts
from jax import random

init_position, inv_mass_matrix = get_position()

parameters = nuts.new_parameters(inv_mass_matrix)  # has default value of step_size and num_integration_steps
init_state = nuts.new_state(init_position, logpdf)
nuts_kernel = nuts.kernel(logpdf, parameters)

rng_key = jax.random.PRNGKey(2020)
new_state, info = nuts_kernel(rng_key, init_state)
```

# How to contribute?

1. `pip install -r requirements-dev.txt`
2. Run `make lint` and `make test` before pushing on the repo; CI should pass if
   these pass locally.
