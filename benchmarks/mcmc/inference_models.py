from inference_gym import using_jax as gym
import jax.numpy as jnp


class SimpleModel(gym.targets.Model):

    def __init__(self, ndims):
      super(SimpleModel, self).__init__(
          default_event_space_bijector=lambda x: x,
          event_shape=(ndims,),
          dtype=jnp.float32,
          name='simple_model',
          pretty_name='SimpleModel',
          sample_transformations=dict(
              identity=gym.targets.Model.SampleTransformation(
                  fn=lambda x: x,
                  pretty_name='Identity',
                  ground_truth_mean=jnp.zeros(ndims),
                  # Variance of Chi2 with one degree of freedom is 2.
                  ground_truth_standard_deviation=jnp.ones(ndims)
              ),
              
              ),
      )

    def _unnormalized_log_prob(self, value):
      return -0.5 * jnp.sum(jnp.square(value))


# square=gym.targets.Model.SampleTransformation(
#                   fn=lambda x: x**2,
#                   pretty_name='x^2',
#                   ground_truth_mean=jnp.array([100.0, 19.0]),
#                   # Variance of Chi2 with one degree of freedom is 2.
#                   ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))
#               ),

models = {}
for target_name in gym.targets.__all__:
  if target_name in ['Banana', 'IllConditionedGaussian']:
  # if target_name in ['Banana']:
    try:
      models[target_name] = getattr(gym.targets, target_name)()
    except:
      pass

models['simple'] = SimpleModel(2)

