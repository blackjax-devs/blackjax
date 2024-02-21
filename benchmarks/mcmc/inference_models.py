from inference_gym import using_jax as gym
import jax
import jax.numpy as jnp


class Normal(gym.targets.Model):

    def __init__(self, ndims):
      super(Normal, self).__init__(
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
      self.E_x2 = jnp.ones(ndims)
      self.Var_x2 = 2 * self.E_x2

    def _unnormalized_log_prob(self, value):
      return -0.5 * jnp.sum(jnp.square(value))

class Banana(gym.targets.Banana):
  def __init__(self):
      super(Banana, self).__init__()
    
      self.E_x2 = jnp.array([100.0, 19.0])
      self.Var_x2 = jnp.array([20000.0, 4600.898])

class IllConditionedGaussian():
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""


    def __init__(self, d, condition_number, numpy_seed=None, prior= 'prior'):
        """numpy_seed is used to generate a random rotation for the covariance matrix.
            If None, the covariance matrix is diagonal."""

        self.name = 'icg'
        self.d = d
        self.condition_number = condition_number
        eigs = jnp.logspace(-0.5 * jnp.log10(condition_number), 0.5 * jnp.log10(condition_number), d)

        if numpy_seed == None:  # diagonal
            self.E_x2 = eigs
            self.R = jnp.eye(d)
            self.Hessian = jnp.diag(1 / eigs)
            self.Cov = jnp.diag(eigs)

        else:  # randomly rotate
            rng = np.random.RandomState(seed=numpy_seed)
            D = jnp.diag(eigs)
            inv_D = jnp.diag(1 / eigs)
            R, _ = jnp.array(np.linalg.qr(rng.randn(self.d, self.d)))  # random rotation
            self.R = R
            self.Hessian = R @ inv_D @ R.T
            self.Cov = R @ D @ R.T
            self.E_x2 = jnp.diagonal(R @ D @ R.T)

        self.Var_x2 = 2 * jnp.square(self.E_x2)


        self.nlogp = lambda x: 0.5 * x.T @ self.Hessian @ x
        self.transform = lambda x: x
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


        if prior == 'map':
            self.prior_draw = lambda key: jnp.zeros(self.d)

        elif prior == 'posterior':
            self.prior_draw = lambda key: self.R @ (jax.random.normal(key, shape=(self.d,)) * jnp.sqrt(eigs))

        else: # N(0, sigma_true_max)
            self.prior_draw = lambda key: jax.random.normal(key, shape=(self.d,)) * jnp.max(jnp.sqrt(eigs))


# square=gym.targets.Model.SampleTransformation(
#                   fn=lambda x: x**2,
#                   pretty_name='x^2',
#                   ground_truth_mean=jnp.array([100.0, 19.0]),
#                   # Variance of Chi2 with one degree of freedom is 2.
#                   ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))
#               ),

# models = {}
# for target_name in gym.targets.__all__:
#   if target_name in ['Banana', 'IllConditionedGaussian']:
#   # if target_name in ['Banana']:
#     try:
#       models[target_name] = getattr(gym.targets, target_name)()
#     except:
#       pass

models = {'normal': Normal(10), 'banana': Banana()}

