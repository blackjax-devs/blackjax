import jax.numpy as jnp
from typing import NamedTuple
import jax

class RobnikStepSizeTuningState(NamedTuple):
    time : jnp.ndarray
    step_size: float
    x_average: float
    step_size_max: float
    num_dimensions: int

def robnik_step_size_tuning(desired_energy_var, trust_in_estimate=1.5, num_effective_samples=150, step_size_max=jnp.inf, step_size_reduction_factor=0.8):
      
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def init(initial_step_size, num_dimensions):
        return RobnikStepSizeTuningState(time=0.0, x_average=0.0, step_size=initial_step_size, step_size_max=step_size_max, num_dimensions=num_dimensions)
      
    def update(robnik_state, info):

        # jax.debug.print("robnik state: {x}", x=robnik_state)
        # jax.debug.print("info: {x}", x=(info.energy_change, info.nonans))
        # raise Exception("Stop here")



        energy_change = info.energy_change


        xi = (
            jnp.square(energy_change) / (robnik_state.num_dimensions * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger on much smaller than the desired one.

        x_average = decay_rate * robnik_state.x_average + weight * (
            xi / jnp.power(robnik_state.step_size, 6.0)
        )
        
        time = decay_rate * robnik_state.time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < robnik_state.step_size_max) * step_size + (
            step_size > robnik_state.step_size_max
        ) * robnik_state.step_size_max  # if the proposed stepsize is above the stepsize where we have seen divergences
        # jax.debug.print("new step_size: {x}", x=(step_size))

        # old_robnik_state = robnik_state

        # old_robnik_step_size = robnik_state.step_size
        # jax.debug.print("step_size: {x}", x=(old_robnik_step_size, old_robnik_step_size * step_size_reduction_factor, step_size))
        # jax.debug.print("stuff {x}", x=(x_average, time, robnik_state.time, time))
        old_robnik_state = robnik_state


        
        robnik_state = jax.lax.cond(
            info.nonans,
            lambda: RobnikStepSizeTuningState(time=time, x_average=x_average, step_size=step_size, step_size_max=step_size_max, num_dimensions=robnik_state.num_dimensions),
            lambda: robnik_state._replace(step_size=robnik_state.step_size * step_size_reduction_factor),
        )
        # jax.debug.print("robnik_state: {robnik_state}", robnik_state=(robnik_state.step_size, info.nonans, robnik_state.step_size * step_size_reduction_factor))
        # jax.debug.print("robnik_state: {x}", x=(old_robnik_state.step_size, robnik_state.step_size))
        return robnik_state

        # return RobnikStepSizeTuningState(time=time, x_average=x_average, step_size=step_size, step_size_max=step_size_max, num_dimensions=robnik_state.num_dimensions)


    def final(robnik_state):
        return robnik_state.step_size

    return init, update, final
