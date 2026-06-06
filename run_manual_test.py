import jax
import jax.numpy as jnp
from tests.test_slingshot_benchmarks import make_linear_regression, run_benchmark_logic

_, logdensity_fn, initial_positions, _ = make_linear_regression()
run_benchmark_logic(logdensity_fn, initial_positions, dim=3)
