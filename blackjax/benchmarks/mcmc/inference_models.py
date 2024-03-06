from inference_gym import using_jax as gym
import jax
import jax.numpy as jnp
import numpy as np
import os
dirr = os.path.dirname(os.path.realpath(__file__))



# class Normal():

#     def __init__(self, ndims):
#       # super(Normal, self).__init__(
#       #     default_event_space_bijector=lambda x: x,
#       #     event_shape=(ndims,),
#       #     dtype=jnp.float32,
#       #     name='simple_model',
#       #     pretty_name='SimpleModel',
#       #     sample_transformations=dict(
#       #         identity=gym.targets.Model.SampleTransformation(
#       #             fn=lambda x: x,
#       #             pretty_name='Identity',
#       #             ground_truth_mean=jnp.zeros(ndims),
#       #             # Variance of Chi2 with one degree of freedom is 2.
#       #             ground_truth_standard_deviation=jnp.ones(ndims)
#       #         ),
              
#       #         ),
#       # )
#       self.ndims = ndims
#       self.name = 'normal'
#       self.E_x2 = jnp.ones(ndims)
#       self.Var_x2 = 2 * self.E_x2

#     def unnormalized_log_prob(self, value):
#       return -0.5 * jnp.sum(jnp.square(value))

# class Banana(gym.targets.Banana):
#   def __init__(self):
#       super(Banana, self).__init__()
    
#       self.ndims = 2
#       self.E_x2 = jnp.array([100.0, 19.0])
#       self.Var_x2 = jnp.array([20000.0, 4600.898])

# class IllConditionedGaussian():
#     """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""


#     def __init__(self, d, condition_number, numpy_seed=None, prior= 'prior'):
#         """numpy_seed is used to generate a random rotation for the covariance matrix.
#             If None, the covariance matrix is diagonal."""

#         self.name = 'icg'
#         self.ndims = d
#         self.condition_number = condition_number
#         eigs = jnp.logspace(-0.5 * jnp.log10(condition_number), 0.5 * jnp.log10(condition_number), d)

#         if numpy_seed == None:  # diagonal
#             self.E_x2 = eigs
#             self.R = jnp.eye(d)
#             self.Hessian = jnp.diag(1 / eigs)
#             self.Cov = jnp.diag(eigs)

#         else:  # randomly rotate
#             rng = np.random.RandomState(seed=numpy_seed)
#             D = jnp.diag(eigs)
#             inv_D = jnp.diag(1 / eigs)
#             R, _ = jnp.array(np.linalg.qr(rng.randn(self.d, self.d)))  # random rotation
#             self.R = R
#             self.Hessian = R @ inv_D @ R.T
#             self.Cov = R @ D @ R.T
#             self.E_x2 = jnp.diagonal(R @ D @ R.T)

#         self.Var_x2 = 2 * jnp.square(self.E_x2)


#         self.unnormalized_log_prob = lambda x: - 0.5 * x.T @ self.Hessian @ x
#         self.transform = lambda x: x
#         # self.grad_nlogp = jax.value_and_grad(self.unnormalized_log_prob)


#         if prior == 'map':
#             self.sample = lambda key: jnp.zeros(self.d)

#         elif prior == 'posterior':
#             self.sample = lambda key: self.R @ (jax.random.normal(key, shape=(self.d,)) * jnp.sqrt(eigs))

#         else: # N(0, sigma_true_max)
#             self.sample = lambda key: jax.random.normal(key, shape=(self.d,)) * jnp.max(jnp.sqrt(eigs))


# # square=gym.targets.Model.SampleTransformation(
# #                   fn=lambda x: x**2,
# #                   pretty_name='x^2',
# #                   ground_truth_mean=jnp.array([100.0, 19.0]),
# #                   # Variance of Chi2 with one degree of freedom is 2.
# #                   ground_truth_standard_deviation=jnp.sqrt(jnp.array([20000.0, 4600.898]))
# #               ),

# # models = {}
# # for target_name in gym.targets.__all__:
# #   if target_name in ['Banana', 'IllConditionedGaussian']:
# #   # if target_name in ['Banana']:
# #     try:
# #       models[target_name] = getattr(gym.targets, target_name)()
# #     except:
# #       pass

class StandardNormal():
    """Standard Normal distribution in d dimensions"""

    def __init__(self, d):
        self.ndims = d
        self.E_x2 = jnp.ones(d)
        self.Var_x2 = 2 * self.E_x2
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.name= 'stn'

    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x), axis= -1)


    def transform(self, x):
        return x

    def sample(self, key):
        return jax.random.normal(key, shape = (self.ndims, ))



class IllConditionedGaussian():
    """Gaussian distribution. Covariance matrix has eigenvalues equally spaced in log-space, going from 1/condition_bnumber^1/2 to condition_number^1/2."""


    def __init__(self, d, condition_number, numpy_seed=None, prior= 'prior'):
        """numpy_seed is used to generate a random rotation for the covariance matrix.
            If None, the covariance matrix is diagonal."""

        self.name = 'icg'
        self.ndims = d
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
            R, _ = jnp.array(np.linalg.qr(rng.randn(self.ndims, self.ndims)))  # random rotation
            self.R = R
            self.Hessian = R @ inv_D @ R.T
            self.Cov = R @ D @ R.T
            self.E_x2 = jnp.diagonal(R @ D @ R.T)

            #Cov_precond = jnp.diag(1 / jnp.sqrt(self.E_x2)) @ self.Cov @ jnp.diag(1 / jnp.sqrt(self.E_x2))

            #print(jnp.linalg.cond(Cov_precond) / jnp.linalg.cond(self.Cov))

        self.Var_x2 = 2 * jnp.square(self.E_x2)


        self.nlogp = lambda x: 0.5 * x.T @ self.Hessian @ x
        self.transform = lambda x: x
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


        if prior == 'map':
            self.sample = lambda key: jnp.zeros(self.ndims)

        elif prior == 'posterior':
            self.sample = lambda key: self.R @ (jax.random.normal(key, shape=(self.ndims,)) * jnp.sqrt(eigs))

        else: # N(0, sigma_true_max)
            self.sample = lambda key: jax.random.normal(key, shape=(self.ndims,)) * jnp.max(jnp.sqrt(eigs))



class IllConditionedESH():
    """ICG from the ESH paper."""

    def __init__(self):
        self.ndims = 50
        self.variance = jnp.linspace(0.01, 1, self.ndims)

        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * jnp.sum(jnp.square(x) / self.variance, axis= -1)


    def transform(self, x):
        return x

    def draw(self, key):
        return jax.random.normal(key, shape = (self.ndims, )) * jnp.sqrt(self.variance)

    def sample(self, key):
        return jax.random.normal(key, shape = (self.ndims, ))




class IllConditionedGaussianGamma():
    """Inference gym's Ill conditioned Gaussian"""

    def __init__(self, prior = 'prior'):
        self.name = 'ICG'
        self.ndims = 100

        # define the Hessian
        rng = np.random.RandomState(seed=10 & (2 ** 32 - 1))
        eigs = np.sort(rng.gamma(shape=0.5, scale=1., size=self.ndims)) #eigenvalues of the Hessian
        eigs *= jnp.average(1.0/eigs)
        self.entropy = 0.5 * self.ndims
        self.maxmin = (1./jnp.sqrt(eigs[0]), 1./jnp.sqrt(eigs[-1])) 
        R, _ = np.linalg.qr(rng.randn(self.ndims, self.ndims)) #random rotation
        self.map_to_worst = (R.T)[[0, -1], :]
        self.Hessian = R @ np.diag(eigs) @ R.T

        # analytic ground truth moments
        self.E_x2 = jnp.diagonal(R @ np.diag(1.0/eigs) @ R.T)
        self.Var_x2 = 2 * jnp.square(self.E_x2)

        # norm = jnp.diag(1/jnp.sqrt(self.E_x2))
        # Sigma = R @ np.diag(1/eigs) @ R.T
        # reduced = norm @ Sigma @ norm
        # print(np.linalg.cond(reduced), np.linalg.cond(Sigma))
        
        # gradient
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        if prior == 'map':
            self.sample = lambda key: jnp.zeros(self.ndims)

        elif prior == 'posterior':
            self.sample = lambda key: R @ (jax.random.normal(key, shape=(self.ndims,)) / jnp.sqrt(eigs))

        else: # N(0, sigma_true_max)
            self.sample = lambda key: jax.random.normal(key, shape=(self.ndims,)) * jnp.max(1.0/jnp.sqrt(eigs))
            
    def nlogp(self, x):
        """- log p of the target distribution"""
        return 0.5 * x.T @ self.Hessian @ x

    def transform(self, x):
        return x
    
    


class Banana():
    """Banana target fromm the Inference Gym"""

    def __init__(self, prior = 'map'):
        self.name = 'Banana'
        self.curvature = 0.03
        self.ndims = 2
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x
        self.E_x2 = jnp.array([100.0, 19.0]) #the first is analytic the second is by drawing 10^8 samples from the generative model. Relative accuracy is around 10^-5.
        self.Var_x2 = jnp.array([20000.0, 4600.898])

        if prior == 'map':
            self.sample = lambda key: jnp.array([0, -100.0 * self.curvature])
        elif prior == 'posterior':
            self.sample = lambda key: self.posterior_draw(key)
        elif prior == 'prior':
            self.sample = lambda key: jax.random.normal(key, shape=(self.ndims,)) * jnp.array([10.0, 5.0]) * 2
        else:
            raise ValueError('prior = '+prior +' is not defined.')

    def nlogp(self, x):
        mu2 = self.curvature * (x[0] ** 2 - 100)
        return 0.5 * (jnp.square(x[0] / 10.0) + jnp.square(x[1] - mu2))

    def posterior_draw(self, key):
        z = jax.random.normal(key, shape = (2, ))
        x0 = 10.0 * z[0]
        x1 = self.curvature * (x0 ** 2 - 100) + z[1]
        return jnp.array([x0, x1])

    def ground_truth(self):
        x = jax.vmap(self.posterior_draw)(jax.random.split(jax.random.PRNGKey(0), 100000000))
        print(jnp.average(x, axis=0))
        print(jnp.average(jnp.square(x), axis=0))
        print(jnp.std(jnp.square(x[:, 0])) ** 2, jnp.std(jnp.square(x[:, 1])) ** 2)

    def plott(self):
        xmin, xmax = -20.0, 20.0
        ymin, ymax = -10.0, 10.0
        X, Y, Z = get_contour_plot(self, jnp.linspace(xmin, xmax, 100), jnp.linspace(ymin, ymax, 100))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.contourf(X, Y, jnp.exp(-Z))

        x = np.linspace(xmin, xmax, 100)
        plt.plot(x, 0.03 * (x ** 2 - 100), color='tab:red')
        plt.savefig('../tst_ensamble/Banana/banana.png')
        plt.show()



class Cauchy():
    """d indpendent copies of the standard Cauchy distribution"""

    def __init__(self, d):
        self.ndims = d

        self.nlogp = lambda x: jnp.sum(jnp.log(1. + jnp.square(x)))
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x        
        self.sample = lambda key: jax.random.normal(key, shape=(self.ndims,))




class HardConvex():

    def __init__(self, d, kappa, theta = 0.1):
        """d is the dimension, kappa = condition number, 0 < theta < 1/4"""
        self.ndims = d
        self.theta, self.kappa = theta, kappa
        C = jnp.power(d-1, 0.25 - theta)
        self.nlogp = lambda x: 0.5 * jnp.sum(jnp.square(x[:-1])) + (0.75 / kappa)* x[-1]**2 - 0.5 * jnp.sum(jnp.cos(C * x[:-1])) / C**2
        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x

        # numerically precomputed variances
        num_integration = [0.93295, 0.968802, 0.990595, 0.998002, 0.999819]
        if d == 100:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[0], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 300:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[1], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 1000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[2], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 3000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[3], jnp.ones(1) * 2.0*kappa/3.0))
        elif d == 10000:
            self.variance = jnp.concatenate((jnp.ones(d-1) * num_integration[4], jnp.ones(1) * 2.0*kappa/3.0))
        else:
            None


    def sample(self, key):
        """Gaussian prior with approximately estimating the variance along each dimension"""
        scale = jnp.concatenate((jnp.ones(self.ndims-1), jnp.ones(1) * jnp.sqrt(2.0 * self.kappa / 3.0)))
        return jax.random.normal(key, shape=(self.ndims,)) * scale




class BiModal():
    """A Gaussian mixture p(x) = f N(x | mu1, sigma1) + (1-f) N(x | mu2, sigma2)."""

    def __init__(self, d = 50, mu1 = 0.0, mu2 = 8.0, sigma1 = 1.0, sigma2 = 1.0, f = 0.2):

        self.ndims = d

        self.mu1 = jnp.insert(jnp.zeros(d-1), 0, mu1)
        self.mu2 = jnp.insert(jnp.zeros(d - 1), 0, mu2)
        self.sigma1, self.sigma2 = sigma1, sigma2
        self.f = f
        self.variance = jnp.insert(jnp.ones(d-1) * ((1 - f) * sigma1**2 + f * sigma2**2), 0, (1-f)*(sigma1**2 + mu1**2) + f*(sigma2**2 + mu2**2))
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""

        N1 = (1.0 - self.f) * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu1), axis= -1) / self.sigma1 ** 2) / jnp.power(2 * jnp.pi * self.sigma1 ** 2, self.ndims * 0.5)
        N2 = self.f * jnp.exp(-0.5 * jnp.sum(jnp.square(x - self.mu2), axis= -1) / self.sigma2 ** 2) / jnp.power(2 * jnp.pi * self.sigma2 ** 2, self.ndims * 0.5)

        return -jnp.log(N1 + N2)


    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.ndims))
        mask = np.random.uniform(0, 1, num_samples) < self.f
        X[mask, :] = (X[mask, :] * self.sigma2) + self.mu2
        X[~mask] = (X[~mask] * self.sigma1) + self.mu1

        return X


    def transform(self, x):
        return x

    def sample(self, key):
        z = jax.random.normal(key, shape = (self.ndims, )) *self.sigma1
        #z= z.at[0].set(self.mu1 + z[0])
        return z


class BiModalEqual():
    """Mixture of two Gaussians, one centered at x = [mu/2, 0, 0, ...], the other at x = [-mu/2, 0, 0, ...].
        Both have equal probability mass."""

    def __init__(self, d, mu):

        self.ndims = d
        self.mu = mu
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""

        return 0.5 * jnp.sum(jnp.square(x), axis= -1) - jnp.log(jnp.cosh(0.5*self.mu*x[0])) + 0.5* self.ndims * jnp.log(2 * jnp.pi) + self.mu**2 / 8.0


    def draw(self, num_samples):
        """direct sampler from a target"""
        X = np.random.normal(size = (num_samples, self.ndims))
        mask = np.random.uniform(0, 1, num_samples) < 0.5
        X[mask, 0] += 0.5*self.mu
        X[~mask, 0] -= 0.5 * self.mu

        return X

    def transform(self, x):
        return x


class Funnel():
    """Noise-less funnel"""

    def __init__(self, d = 20):

        self.ndims = d
        self.sigma_theta= 3.0
        self.variance = jnp.ones(d)
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        X = x[..., :- 1]

        return 0.5* jnp.square(theta / self.sigma_theta) + 0.5 * (self.ndims - 1) * theta + 0.5 * jnp.exp(-theta) * jnp.sum(jnp.square(X), axis = -1)

    def inverse_transform(self, xtilde):
        theta = 3 * xtilde[-1]
        return jnp.concatenate((xtilde[:-1] * jnp.exp(0.5 * theta), jnp.ones(1)*theta))


    def transform(self, x):
        """gaussianization"""
        xtilde = jnp.empty(x.shape)
        xtilde = xtilde.at[-1].set(x.T[-1] / 3.0)
        xtilde = xtilde.at[:-1].set(x.T[:-1] * jnp.exp(-0.5*x.T[-1]))
        return xtilde.T


    def sample(self, key):
        return self.inverse_transform(jax.random.normal(key, shape = (self.ndims, )))




class Funnel_with_Data():

    def __init__(self, d, sigma, minibatch_size, key):

        self.ndims = d
        self.sigma_theta= 3.0
        self.theta_true = 0.0
        self.sigma_data = sigma
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

        self.data = self.simulate_data()

        self.batch = minibatch_size

    def simulate_data(self):

        norm = jax.random.normal(jax.random.PRNGKey(123), shape = (2*(self.ndims-1), ))
        z_true = norm[:self.ndims-1] * jnp.exp(self.theta_true * 0.5)
        self.data = z_true + norm[self.ndims-1:] * self.sigma_data


    def nlogp(self, x, subset):
        """ - log p of the target distribution
                x = [z_0, z_1, ... z_{d-1}, theta] """
        theta = x[-1]
        z = x[:- 1][subset]

        prior_theta = jnp.square(theta / self.sigma_theta)
        prior_z = jnp.sum(subset) * theta + jnp.exp(-theta) * jnp.sum(jnp.square(z*subset))
        likelihood = jnp.sum(jnp.square((z - self.data)*subset / self.sigma_data))

        return 0.5 * (prior_theta + prior_z + likelihood)


    def transform(self, x):
        """gaussianization"""
        return x

    def sample(self, key):
        key1, key2 = jax.random.split(key)
        theta = jax.random.normal(key1) * self.sigma_theta
        z = jax.random.normal(key2, shape = (self.ndims-1, )) * jnp.exp(theta * 0.5)
        return jnp.concatenate((z, theta))




class Rosenbrock():

    def __init__(self, d = 36, Q = 0.1):

        self.ndims = d
        self.Q = Q
        self.name = 'rosenbrock'
        #ground truth moments
        var_x = 2.0

        #these two options were precomputed:
        if Q == 0.1:
            var_y = 10.098433122783046 # var_y is computed numerically (see class function compute_variance)
        elif Q == 0.5:
            var_y = 10.498957879911487
        else:
            raise ValueError('Ground truth moments for Q = ' + str(Q) + ' were not precomputed. Use Q = 0.1 or 0.5.')

        self.variance = jnp.concatenate((var_x * jnp.ones(d//2), var_y * jnp.ones(d//2)))

        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution"""
        X, Y = x[..., :self.ndims//2], x[..., self.ndims//2:]
        return 0.5 * jnp.sum(jnp.square(X - 1.0) + jnp.square(jnp.square(X) - Y) / self.Q, axis= -1)



    def draw(self, num):
        n = self.ndims // 2
        X= np.empty((num, self.ndims))
        X[:, :n] = np.random.normal(loc= 1.0, scale= 1.0, size= (num, n))
        X[:, n:] = np.random.normal(loc= jnp.square(X[:, :n]), scale= jnp.sqrt(self.Q), size= (num, n))

        return X


    def transform(self, x):
        return x


    def sample(self, key):
        return jax.random.normal(key, shape = (self.ndims, ))


    def ground_truth(self):
        num = 100000000
        x = np.random.normal(loc=1.0, scale=1.0, size=num)
        y = np.random.normal(loc=np.square(x), scale=jnp.sqrt(self.Q), size=num)

        x2 = jnp.sum(jnp.square(x)) / (num - 1)
        y2 = jnp.sum(jnp.square(y)) / (num - 1)

        x1 = np.average(x)
        y1 = np.average(y)

        print(np.sqrt(0.5*(np.square(np.std(x)) + np.square(np.std(y)))))

        print(x2, y2)



class Brownian():
    """
    log sigma_i ~ N(0, 2)
    log sigma_obs ~N(0, 2)

    x ~ RandomWalk(0, sigma_i)
    x_observed = (x + noise) * mask
    noise ~ N(0, sigma_obs)
    mask = 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    """

    def __init__(self):
        self.num_data = 30
        self.ndims = self.num_data + 2
        self.name = 'brownian'

        ground_truth_moments = jnp.load(dirr + '/ground_truth/' + self.name + '/ground_truth.npy')
        self.E_x2, self.Var_x2 = ground_truth_moments[0], ground_truth_moments[1]

        self.data = jnp.array([0.21592641, 0.118771404, -0.07945447, 0.037677474, -0.27885845, -0.1484156, -0.3250906, -0.22957903,
                               -0.44110894, -0.09830782, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8786016, -0.83736074,
                               -0.7384849, -0.8939254, -0.7774566, -0.70238715, -0.87771565, -0.51853573, -0.6948214, -0.6202789])
        # sigma_obs = 0.15, sigma_i = 0.1

        self.observable = jnp.concatenate((jnp.ones(10), jnp.zeros(10), jnp.ones(10)))
        self.num_observable = jnp.sum(self.observable)  # = 20
        self.grad_nlogp = jax.value_and_grad(self.nlogp)

    def nlogp(self, x):
        # y = softplus_to_log(x[:2])

        lik = 0.5 * jnp.exp(-2 * x[1]) * jnp.sum(self.observable * jnp.square(x[2:] - self.data)) + x[
            1] * self.num_observable
        prior_x = 0.5 * jnp.exp(-2 * x[0]) * (x[2] ** 2 + jnp.sum(jnp.square(x[3:] - x[2:-1]))) + x[0] * self.num_data
        prior_logsigma = 0.5 * jnp.sum(jnp.square(x / 2.0))

        return lik + prior_x + prior_logsigma


    def transform(self, x):
        return jnp.concatenate((jnp.exp(x[:2]), x[2:]))

    # def sample(self, key):
    #     """draws x from the prior"""

    #     return jax.scipy.optimize.minimize(self.nlogp, x0 = jnp.zeros(self.d), method = 'BFGS', options = {'maxiter': 100}).x

    def sample(self, key):
        key_walk, key_sigma = jax.random.split(key)

        # original prior
        # log_sigma = jax.random.normal(key_sigma, shape= (2, )) * 2

        # narrower prior
        log_sigma = jnp.log(np.array([0.1, 0.15])) + jax.random.normal(key_sigma, shape=(
        2,)) * 0.1  # *0.05# log sigma_i, log sigma_obs

        walk = random_walk(key_walk, self.ndims - 2) * jnp.exp(log_sigma[0])

        return jnp.concatenate((log_sigma, walk))

    def generate_data(self, key):
        key_walk, key_sigma, key_noise = jax.random.split(key, 3)

        log_sigma = jax.random.normal(key_sigma, shape=(2,)) * 2  # log sigma_i, log sigma_obs

        walk = random_walk(key_walk, self.ndims - 2) * jnp.exp(log_sigma[0])
        noise = jax.random.normal(key_noise, shape=(self.ndims - 2,)) * jnp.exp(log_sigma[1])

        return walk + noise


class GermanCredit:
    """ Taken from inference gym.

        x = (global scale, local scales, weights)

        global_scale ~ Gamma(0.5, 0.5)

        for i in range(num_features):
            unscaled_weights[i] ~ Normal(loc=0, scale=1)
            local_scales[i] ~ Gamma(0.5, 0.5)
            weights[i] = unscaled_weights[i] * local_scales[i] * global_scale

        for j in range(num_datapoints):
            label[j] ~ Bernoulli(features @ weights)

        We use a log transform for the scale parameters.
    """

    def __init__(self):
        self.ndims = 51 #global scale + 25 local scales + 25 weights
        self.name = 'GC'

        self.labels = jnp.load(dirr + '/data/gc_labels.npy')
        self.features = jnp.load(dirr + '/data/gc_features.npy')

        truth = jnp.load(dirr+'/ground_truth/' + self.name + '/ground_truth.npy')
        self.E_x2, self.Var_x2 = truth[0], truth[1]

        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def transform(self, x):
        return jnp.concatenate((jnp.exp(x[:26]), x[26:]))

    def nlogp(self, x):

        scales = jnp.exp(x[:26])

        # prior
        pr = jnp.sum(0.5 * scales + 0.5 * x[:26]) + 0.5 * jnp.sum(jnp.square(x[26:]))

        # transform
        transform = -jnp.sum(x[:26])

        # likelihood
        weights = scales[0] * scales[1:26] * x[26:]
        logits = self.features @ weights # = jnp.einsum('nd,...d->...n', self.features, weights)
        lik = jnp.sum(self.labels * jnp.logaddexp(0., -logits) + (1-self.labels)* jnp.logaddexp(0., logits))

        return lik + pr + transform
    #
    # def sample(self, key):
    #     key1, key2 = jax.random.split(key)
    #
    #     scales = jax.random.gamma(key1, 0.5, shape=(26,)) * 2.  # we divided by beta = 0.5
    #     unscaled_weights = jax.random.normal(key2, shape=(25,))
    #
    #     return jnp.concatenate((scales, unscaled_weights))
    #

    def sample(self, key):
        weights = jax.random.normal(key, shape = (25, ))
        return jnp.concatenate((jnp.zeros(26), weights))




class ItemResponseTheory:
    """ Taken from inference gym."""

    def __init__(self):
        self.ndims = 501
        self.name = 'IRT'
        self.students = 400
        self.questions = 100

        self.mask = jnp.load(dirr + '/data/irt_mask.npy')
        self.labels = jnp.load(dirr + '/data/irt_labels.npy')

        truth = jnp.load(dirr+'/ground_truth/' + self.name + '/ground_truth.npy')
        self.E_x2, self.Var_x2 = truth[0], truth[1]

        self.grad_nlogp = jax.value_and_grad(self.nlogp)
        self.transform = lambda x: x

    def nlogp(self, x):

        students = x[:self.students]
        mean = x[self.students]
        questions = x[self.students + 1:]

        # prior
        pr = 0.5 * (jnp.square(mean - 0.75) + jnp.sum(jnp.square(students)) + jnp.sum(jnp.square(questions)))

        # likelihood
        logits = mean + students[:, jnp.newaxis] - questions[jnp.newaxis, :]
        bern = self.labels * jnp.logaddexp(0., -logits) + (1 - self.labels) * jnp.logaddexp(0., logits)
        bern = jnp.where(self.mask, bern, jnp.zeros_like(bern))
        lik = jnp.sum(bern)

        return lik + pr


    def sample(self, key):
        x = jax.random.normal(key, shape = (self.ndims,))
        x = x.at[self.students].add(0.75)
        return x




class StochasticVolatility():
    """Example from https://num.pyro.ai/en/latest/examples/stochastic_volatility.html"""

    def __init__(self):
        self.SP500_returns = jnp.load(dirr + '/data/SP500.npy')

        self.name = 'SV'
        self.ndims = 2429

        self.typical_sigma, self.typical_nu = 0.02, 10.0 # := 1 / lambda

        data = jnp.load(dirr + '/ground_truth/stochastic_volatility/ground_truth_0.npy')
        self.E_x2 = data[0]
        self.Var_x2 = data[1]
        self.grad_nlogp = jax.value_and_grad(self.nlogp)


    def nlogp(self, x):
        """- log p of the target distribution
            x=  [s1, s2, ... s2427, log sigma / typical_sigma, log nu / typical_nu]"""

        sigma = jnp.exp(x[-2]) * self.typical_sigma #we used this transformation to make x unconstrained
        nu = jnp.exp(x[-1]) * self.typical_nu

        l1= (jnp.exp(x[-2]) - x[-2]) + (jnp.exp(x[-1]) - x[-1])
        l2 = (self.ndims - 2) * jnp.log(sigma) + 0.5 * (jnp.square(x[0]) + jnp.sum(jnp.square(x[1:-2] - x[:-3]))) / jnp.square(sigma)
        l3 = jnp.sum(nlogp_StudentT(self.SP500_returns, nu, jnp.exp(x[:-2])))

        return l1 + l2 + l3


    def transform(self, x):
        """transforms to the variables which are used by numpyro (and in which we have the ground truth moments)"""

        z = jnp.empty(x.shape)
        z = z.at[:-2].set(x[:-2]) # = s = log R
        z = z.at[-2].set(jnp.exp(x[-2]) * self.typical_sigma) # = sigma
        z = z.at[-1].set(jnp.exp(x[-1]) * self.typical_nu) # = nu

        return z


    def sample(self, key):
        """draws x from the prior"""

        key_walk, key_exp = jax.random.split(key)

        scales = jnp.array([self.typical_sigma, self.typical_nu])
        #params = jax.random.exponential(key_exp, shape = (2, )) * scales
        params= scales
        walk = random_walk(key_walk, self.ndims - 2) * params[0]
        return jnp.concatenate((walk, jnp.log(params/scales)))
    




def nlogp_StudentT(x, df, scale):
    y = x / scale
    z = (
        jnp.log(scale)
        + 0.5 * jnp.log(df)
        + 0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln(0.5 * df)
        - jax.scipy.special.gammaln(0.5 * (df + 1.0))
    )
    return 0.5 * (df + 1.0) * jnp.log1p(y**2.0 / df) + z



def random_walk(key, num):
    """ Genereting process for the standard normal walk:
        x[0] ~ N(0, 1)
        x[n+1] ~ N(x[n], 1)

        Args:
            key: jax random key
            num: number of points in the walk
        Returns:
            1 realization of the random walk (array of length num)
    """

    def step(track, useless):
        x, key = track
        randkey, subkey = jax.random.split(key)
        x += jax.random.normal(subkey)
        return (x, randkey), x

    return jax.lax.scan(step, init=(0.0, key), xs=None, length=num)[1]



class DiagonalPreconditioned():
    """A target instance which takes some other target and preconditions it"""

    def __init__(self, Target, a):
        """ Target: original target
            a: scaling vector (sqrt(variance of x)), such that x' = x / a"""

        self.Target = Target
        self.ndims= Target.d
        self.a = a
        self.variance = Target.variance / jnp.square(a)


    def nlogp(self, x):
        return self.Target.nlogp(self.a * x)

    def grad_nlogp(self, x):
        return self.Target.grad_nlogp(self.a * x) * self.a

    def draw(self, num):
        return self.Target.draw(num) / self.a

    def transform(self, x):
        return x


models = {'normal': StandardNormal(10), 'banana': Banana(), 'icg' : IllConditionedGaussian(10, 2), }

