import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import blackjax
from blackjax.mcmc.slingshot import init_adaptation, dual_averaging_step

def make_linear_regression():
    key = jax.random.PRNGKey(42)
    N, D = 500, 3
    X = jax.random.normal(key, (N, D))
    true_beta = jnp.array([1.5, -2.0, 0.8])
    true_sigma = 0.5
    y = X @ true_beta + true_sigma * jax.random.normal(jax.random.PRNGKey(43), (N,))
    
    def logdensity(theta):
        beta = theta[:3]
        log_sigma = theta[3]
        sigma = jnp.exp(log_sigma)
        mu = X @ beta
        log_lik = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((y - mu) / sigma)**2)
        log_prior_beta = jnp.sum(-0.5 * beta**2)
        log_prior_sigma = -0.5 * log_sigma**2
        return log_lik + log_prior_beta + log_prior_sigma
        
    initial_positions = jnp.zeros((16, 4))
    true_params = jnp.concatenate([true_beta, jnp.array([jnp.log(true_sigma)])])
    return "1. Linear Regression", logdensity, initial_positions, true_params

def make_logit_regression():
    key = jax.random.PRNGKey(101)
    N, D = 500, 3
    X = jax.random.normal(key, (N, D))
    true_beta = jnp.array([-1.0, 2.5, 0.5])
    logits = X @ true_beta
    probs = jax.nn.sigmoid(logits)
    y = jax.random.bernoulli(jax.random.PRNGKey(102), probs).astype(jnp.float32)
    
    def logdensity(beta):
        logits_pred = X @ beta
        log_lik = jnp.sum(y * jax.nn.log_sigmoid(logits_pred) + (1 - y) * jax.nn.log_sigmoid(-logits_pred))
        log_prior = jnp.sum(-0.5 * beta**2)
        return log_lik + log_prior
        
    initial_positions = jnp.zeros((16, 3))
    return "2. Logit Regression", logdensity, initial_positions, true_beta

def make_hierarchical_model():
    J, M = 8, 20
    key = jax.random.PRNGKey(201)
    key_grp, key_obs = jax.random.split(key)
    
    true_mu = 2.0
    true_tau = 0.8
    true_sigma = 1.0
    
    alphas = true_mu + true_tau * jax.random.normal(key_grp, (J,))
    
    group_indices = jnp.repeat(jnp.arange(J), M)
    y = alphas[group_indices] + true_sigma * jax.random.normal(key_obs, (J * M,))
    
    def logdensity(theta):
        mu_global = theta[0]
        log_tau_global = theta[1]
        alphas_eval = theta[2:10]
        log_sigma_res = theta[10]
        
        tau = jnp.exp(log_tau_global)
        sigma = jnp.exp(log_sigma_res)
        
        log_prior_mu = -0.5 * mu_global**2
        log_prior_tau = -0.5 * log_tau_global**2
        log_prior_sigma = -0.5 * log_sigma_res**2
        
        log_prior_alphas = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * tau**2) - 0.5 * ((alphas_eval - mu_global) / tau)**2)
        
        mu_obs = alphas_eval[group_indices]
        log_lik = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * sigma**2) - 0.5 * ((y - mu_obs) / sigma)**2)
        
        return log_prior_mu + log_prior_tau + log_prior_sigma + log_prior_alphas + log_lik

    initial_positions = jnp.zeros((16, 11))
    true_params = jnp.concatenate([jnp.array([true_mu, jnp.log(true_tau)]), alphas, jnp.array([jnp.log(true_sigma)])])
    return "3. Hierarchical Model", logdensity, initial_positions, true_params

def make_neals_funnel():
    def logdensity(theta):
        v = theta[0]
        x = theta[1:10]
        
        log_prior_v = -0.5 * jnp.log(2 * jnp.pi * 9.0) - 0.5 * (v**2 / 9.0)
        
        var_x = jnp.exp(v)
        log_prior_x = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * var_x) - 0.5 * (x**2 / var_x))
        
        return log_prior_v + log_prior_x

    initial_positions = jax.random.normal(jax.random.PRNGKey(301), (16, 10))
    true_params = jnp.zeros(10)
    return "4. Neal's Funnel", logdensity, initial_positions, true_params

def make_correlated_gaussian():
    dim = 5
    cov = jnp.eye(dim) + 0.5 * (jnp.ones((dim, dim)) - jnp.eye(dim))
    inv_cov = jnp.linalg.inv(cov)
    
    def logdensity(theta):
        return -0.5 * jnp.dot(theta, jnp.dot(inv_cov, theta))
        
    initial_positions = jax.random.normal(jax.random.PRNGKey(401), (16, dim))
    true_params = jnp.zeros(dim)
    return "5. Correlated Gaussian", logdensity, initial_positions, true_params

def run_benchmark(name, logdensity_fn, initial_positions, true_params):
    num_chains = 16
    num_proposals = 1000
    num_warmup = 1000
    num_steps = 1000
    target_rate = 0.65
    dim = initial_positions.shape[-1]
    
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    def init_chain(pos):
        algo = blackjax.slingshot(logdensity_fn, step_size=1.0, num_proposals=num_proposals)
        return algo.init(pos)
    
    # --- CONDITIONAL MAP INITIALIZATION ---
    jitter_key = jax.random.PRNGKey(999)
    jitter = jax.random.normal(jitter_key, initial_positions.shape) * 0.1

    # Bypass the optimizer for models with infinite degenerate spikes
    if "Funnel" in name or "Horseshoe" in name:
        print("Pathological geometry detected. Bypassing MAP and using standard jitter...")
        warm_start_positions = initial_positions + jitter
    else:
        print("Finding MAP estimate for initialization...")
        def neg_log_density(theta):
            return -logdensity_fn(theta)
            
        val_and_grad_fn = jax.jit(jax.value_and_grad(neg_log_density))
        
        def scipy_objective(theta_np):
            val, grad = val_and_grad_fn(jnp.array(theta_np))
            return np.array(val).astype(np.float64), np.array(grad).astype(np.float64)
            
        opt_result = scipy.optimize.minimize(
            scipy_objective, 
            np.array(initial_positions[0]), 
            method="BFGS",
            jac=True
        )
        map_estimate = jnp.array(opt_result.x)
        warm_start_positions = map_estimate + jnp.where(map_estimate == 0, jitter, map_estimate * 0.01)
    # -----------------------------------------------
    
    states = jax.vmap(init_chain)(warm_start_positions)
    
    init_adapt_vmap = jax.vmap(lambda ss: init_adaptation(ss, dim))
    da_states = init_adapt_vmap(jnp.ones(num_chains) * 0.1) 
    
    @jax.jit
    def warmup_step(carry, step_key):
        states, da_states = carry
        keys = jax.random.split(step_key, num_chains)
        
        def single_chain_warmup(key, state, da_state):
            step_size = jnp.exp(da_state.log_step_size) 
            algo = blackjax.slingshot(
                logdensity_fn, 
                step_size=step_size, 
                num_proposals=num_proposals, 
                cholesky=da_state.cholesky
            )
            next_state, info = algo.step(key, state)
            
            acceptance_rate = getattr(info, "acceptance_rate", target_rate)
            next_da_state = dual_averaging_step(
                da_state, 
                acceptance_rate,
                next_state.position,
                target_rate=target_rate
            )
            
            # Step-size floor block to prevent MALA momentum collapse
            min_log_step = jnp.log(0.05)
            next_da_state = next_da_state._replace(
                log_step_size=jnp.maximum(next_da_state.log_step_size, min_log_step),
                log_step_size_bar=jnp.maximum(next_da_state.log_step_size_bar, min_log_step)
            )
            
            return next_state, next_da_state
            
        next_states, next_da_states = jax.vmap(single_chain_warmup)(keys, states, da_states)
        return (next_states, next_da_states), None

    print(f"Running {num_warmup} Warmup Adaptation steps...")
    warmup_keys = jax.random.split(jax.random.PRNGKey(10), num_warmup)
    (states, da_states), _ = jax.lax.scan(warmup_step, (states, da_states), warmup_keys)
    
    final_step_sizes = jnp.exp(da_states.log_step_size_bar)
    print(f"Adapted step sizes (mean across chains): {jnp.mean(final_step_sizes):.4f}")
    final_choleskys = da_states.cholesky
    
    @jax.jit
    def sample_step(carry_states, step_key):
        keys = jax.random.split(step_key, num_chains)
        def single_chain_sample(key, state, step_size, cholesky):
            algo = blackjax.slingshot(
                logdensity_fn, 
                step_size=step_size, 
                num_proposals=num_proposals,
                cholesky=cholesky
            )
            next_state, info = algo.step(key, state)
            return next_state, next_state.position
            
        next_states, positions = jax.vmap(single_chain_sample)(keys, carry_states, final_step_sizes, final_choleskys)
        return next_states, positions

    print(f"Running {num_steps} Production Sampling steps...")
    sample_keys = jax.random.split(jax.random.PRNGKey(11), num_steps)
    _, positions = jax.lax.scan(sample_step, states, sample_keys)
    
    print(f"Sampling completed. Output shape: {positions.shape}")
    mean_recovered = jnp.mean(positions, axis=(0, 1))
    
    print(f"True Params : {true_params}")
    print(f"Recov Params: {mean_recovered}")
    
    mae = jnp.mean(jnp.abs(mean_recovered - true_params))
    print(f"Mean Absolute Error: {mae:.4f}")

if __name__ == "__main__":
    def make_rosenbrock():
        D = 10
        def logdensity(theta):
            term1 = 100.0 * (theta[1:] - theta[:-1]**2)**2
            term2 = (1.0 - theta[:-1])**2
            return -jnp.sum(term1 + term2)
        initial_positions = jax.random.normal(jax.random.PRNGKey(501), (16, D))
        true_params = jnp.ones(D)
        return "6. Rosenbrock Twisted Banana", logdensity, initial_positions, true_params

    def make_horseshoe():
        D = 20
        N = 100
        key = jax.random.PRNGKey(601)
        key_x, key_noise = jax.random.split(key)
        X = jax.random.normal(key_x, (N, D))
        true_beta = jnp.zeros(D)
        true_beta = true_beta.at[jnp.array([2, 7, 15])].set(jnp.array([3.5, -2.1, 4.0]))
        true_sigma = 1.0
        y = X @ true_beta + true_sigma * jax.random.normal(key_noise, (N,))
        
        def logdensity(theta_flat):
            log_tau = theta_flat[0]
            log_lambda = theta_flat[1:21]
            beta = theta_flat[21:41]
            
            tau = jnp.exp(log_tau)
            lambdas = jnp.exp(log_lambda)
            
            def half_cauchy_logpdf_unconstrained(log_val):
                val = jnp.exp(log_val)
                return log_val - jnp.log(1.0 + val**2)
                
            log_prior_tau = half_cauchy_logpdf_unconstrained(log_tau)
            log_prior_lambda = jnp.sum(half_cauchy_logpdf_unconstrained(log_lambda))
            
            var_beta = (tau * lambdas)**2 + 1e-8
            log_prior_beta = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * var_beta) - 0.5 * (beta**2 / var_beta))
            
            mu = X @ beta
            log_lik = jnp.sum(-0.5 * jnp.log(2 * jnp.pi * true_sigma**2) - 0.5 * ((y - mu) / true_sigma)**2)
            
            return log_prior_tau + log_prior_lambda + log_prior_beta + log_lik

        initial_positions = jnp.zeros((16, 41))
        true_params = jnp.concatenate([jnp.array([-1.0]), jnp.zeros(D), true_beta])
        return "7. High-Dimensional Horseshoe", logdensity, initial_positions, true_params

    def make_lgcp():
        M = 4
        D = M * M
        mu_val = 1.0
        sigma_val = 0.5
        l_val = 1.0
        
        coords = jnp.stack(jnp.meshgrid(jnp.arange(M), jnp.arange(M)), axis=-1).reshape(-1, 2)
        dists_sq = jnp.sum((coords[:, None, :] - coords[None, :, :])**2, axis=-1)
        Sigma = sigma_val**2 * jnp.exp(-dists_sq / (2.0 * l_val**2)) + 1e-6 * jnp.eye(D)
        
        L = jnp.linalg.cholesky(Sigma)
        L_inv = jnp.linalg.inv(L)
        Sigma_inv = L_inv.T @ L_inv
        
        key = jax.random.PRNGKey(701)
        key_y, key_counts = jax.random.split(key)
        
        true_Y = mu_val + L @ jax.random.normal(key_y, (D,))
        rates = jnp.exp(true_Y)
        y_counts = jax.random.poisson(key_counts, rates)
        
        def logdensity(Y_flat):
            diff = Y_flat - mu_val
            log_prior = -0.5 * jnp.dot(diff, jnp.dot(Sigma_inv, diff))
            log_lik = jnp.sum(y_counts * Y_flat - jnp.exp(Y_flat))
            return log_prior + log_lik

        initial_positions = jnp.zeros((16, D)) + mu_val
        return "8. Log-Gaussian Cox Process", logdensity, initial_positions, true_Y

    models = [
        make_linear_regression(),
        make_logit_regression(),
        make_hierarchical_model(),
        make_neals_funnel(),
        make_correlated_gaussian(),
        make_rosenbrock(),
        make_horseshoe(),
        make_lgcp()
    ]
    
    for model_args in models:
        run_benchmark(*model_args)