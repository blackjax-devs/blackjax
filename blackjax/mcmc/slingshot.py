import jax
import jax.numpy as jnp
import jax.scipy.linalg
from typing import Callable, NamedTuple

from blackjax.base import SamplingAlgorithm, build_sampling_algorithm

__all__ = ["as_top_level_api", "init", "build_kernel", "SlingshotState", "SlingshotInfo"]

class SlingshotState(NamedTuple):
    """State of the Slingshot MP-MCMC sampler."""
    position: jnp.ndarray
    log_density: float

class SlingshotInfo(NamedTuple):
    """Internal diagnostics for the exact Slingshot transition step."""
    proposal_cloud: jnp.ndarray
    weights: jnp.ndarray
    chosen_index: jnp.ndarray
    acceptance_rate: float
    is_accepted: bool

class SlingshotAdaptState(NamedTuple):
    """State parameters for Nesterov dual-averaging step-size tuning."""
    log_step_size: float
    log_step_size_bar: float
    h_bar: float
    t: int
    mean: jnp.ndarray       # Running mean of accepted positions (shape: (dim,))
    m2: jnp.ndarray         # Running sum of directed outer products (shape: (dim, dim))
    cholesky: jnp.ndarray   # Cholesky factor L of the covariance matrix (shape: (dim, dim))

def init(position: jnp.ndarray, logdensity_fn: Callable, *, rng_key=None, **kwargs) -> SlingshotState:
    """Initialize the Slingshot sampler state from a starting position."""
    return SlingshotState(position=position, log_density=logdensity_fn(position))

def init_adaptation(initial_step_size: float, dim: int) -> SlingshotAdaptState:
    """Initialize dual-averaging and Welford covariance state parameters."""
    log_ss = jnp.log(initial_step_size)
    return SlingshotAdaptState(
        log_step_size=log_ss,
        log_step_size_bar=log_ss,
        h_bar=0.0,
        t=0,
        mean=jnp.zeros(dim),
        m2=jnp.zeros((dim, dim)),
        cholesky=jnp.eye(dim)
    )

def welford_covariance_update(state: SlingshotAdaptState, position: jnp.ndarray, t_global: int) -> SlingshotAdaptState:
    # Window is 150 to 850 (700 steps)
    window_t = t_global - 150 + 1
    
    delta = position - state.mean
    new_mean = state.mean + delta / window_t
    delta2 = position - new_mean
    new_m2 = state.m2 + jnp.outer(delta, delta2)
    
    def compute_cholesky():
        covariance = new_m2 / jnp.maximum(window_t - 1, 1)
        stable_covariance = covariance + 1e-5 * jnp.eye(position.shape[0])
        return jnp.linalg.cholesky(stable_covariance)
        
    new_cholesky = jax.lax.cond(
        t_global == 850,
        compute_cholesky,
        lambda: state.cholesky
    )
    
    return state._replace(mean=new_mean, m2=new_m2, cholesky=new_cholesky)

def dual_averaging_step(
    adapt_state: SlingshotAdaptState,
    acceptance_rate: float,
    position: jnp.ndarray,
    target_rate: float = 0.65,
    gamma: float = 0.05,
    t0: int = 10,
    kappa: float = 0.75
) -> SlingshotAdaptState:
    t = adapt_state.t + 1
    alpha = target_rate - acceptance_rate
    h_bar = (1.0 - 1.0 / (t + t0)) * adapt_state.h_bar + (1.0 / (t + t0)) * alpha
    
    log_step_size = - (jnp.sqrt(t) / gamma) * h_bar
    eta = t ** (-kappa)
    log_step_size_bar = (1.0 - eta) * adapt_state.log_step_size_bar + eta * log_step_size
    
    # Expanded adaptation window matches the 1000-step warmup
    in_window = (t >= 150) & (t <= 850)
    
    next_adapt_state = jax.lax.cond(
        in_window,
        lambda s: welford_covariance_update(s, position, t),
        lambda s: s,
        adapt_state
    )
    
    return next_adapt_state._replace(
        log_step_size=log_step_size,
        log_step_size_bar=log_step_size_bar,
        h_bar=h_bar,
        t=t,
    )

def build_kernel() -> Callable:
    """Build the functional transition kernel for a Gradient-Guided Slingshot sampler (MALA Cloud)."""
    def one_step(
        rng_key: jax.random.PRNGKey,
        state: SlingshotState,
        logdensity_fn: Callable,
        step_size: float,
        num_proposals: int,
        cholesky: jnp.ndarray = None,
    ) -> tuple[SlingshotState, SlingshotInfo]:
        
        key_cloud, key_select, key_accept, key_reverse = jax.random.split(rng_key, 4)
        dim = state.position.shape[0]
        
        # 1. Enable Gradients via JAX
        val_and_grad_fn = jax.value_and_grad(logdensity_fn)
        vmap_val_and_grad = jax.vmap(val_and_grad_fn)
        
        if cholesky is None:
            cholesky = jnp.eye(dim)
            
        covariance = cholesky @ cholesky.T
        
        # --- Helper Functions for Langevin Dynamics ---
        def get_drift(pos, grad):
            """Shift the mean of the proposal cloud along the preconditioned gradient."""
            
            # 1. Calculate the norm of the gradient
            grad_norm = jnp.linalg.norm(grad) + 1e-8
            
            # 2. Safely scale down any massive gradients to a max magnitude of 5.0
            # This preserves the directional momentum while acting as a speed limit
            safe_grad = jnp.where(grad_norm > 5.0, grad * (5.0 / grad_norm), grad)
            
            # 3. Calculate drift using the stabilized gradient
            return pos + (step_size**2 / 2.0) * jnp.dot(covariance, safe_grad)
            
        def log_q(start_pos, end_pos, start_drift):
            """Calculate exact asymmetric transition probability log q(end | start)."""
            diff = end_pos - start_drift
            # Efficient Mahalanobis distance calculation using the lower Cholesky factor L
            inv_L_diff = jax.scipy.linalg.solve_triangular(cholesky, diff, lower=True)
            mahalanobis_sq = jnp.sum(inv_L_diff**2, axis=-1)
            return -0.5 * mahalanobis_sq / (step_size**2)

        # 2. Current State Kinematics
        current_log_dens, current_grad = val_and_grad_fn(state.position)
        current_drift = get_drift(state.position, current_grad)
        
        # 3. Forward Pass: Gradient-Guided Cloud
        raw_noise = jax.random.normal(key_cloud, shape=(num_proposals, dim))
        noise = jnp.dot(raw_noise, cholesky.T)
        proposal_cloud = current_drift + noise * step_size
        
        cloud_log_densities, cloud_grads = vmap_val_and_grad(proposal_cloud)
        cloud_drifts = jax.vmap(get_drift)(proposal_cloud, cloud_grads)
        
        # Calculate the exact Multiple-Try asymmetric weights: w(x, y) = pi(y) * q(x | y)
        def compute_reverse_q(y_i, y_i_drift):
            return log_q(y_i, state.position, y_i_drift)
            
        log_q_x_given_y = jax.vmap(compute_reverse_q)(proposal_cloud, cloud_drifts)
        log_weights = cloud_log_densities + log_q_x_given_y
        
        max_log_weight = jnp.max(log_weights)
        stabilized_weights = jnp.exp(log_weights - max_log_weight)
        sum_forward_weights = jnp.sum(stabilized_weights)
        probabilities = stabilized_weights / sum_forward_weights
        
        # 4. Extract Candidate Proposal
        chosen_index = jax.random.choice(key_select, num_proposals, p=probabilities)
        candidate_position = proposal_cloud[chosen_index]
        candidate_log_density = cloud_log_densities[chosen_index]
        candidate_drift = cloud_drifts[chosen_index]
        
        # 5. Reverse Pass: Validation Cloud
        raw_reverse_noise = jax.random.normal(key_reverse, shape=(num_proposals - 1, dim))
        reverse_noise = jnp.dot(raw_reverse_noise, cholesky.T)
        reverse_cloud_minus_one = candidate_drift + reverse_noise * step_size
        
        # Append the original state to complete the exact reverse cloud
        reverse_cloud = jnp.vstack([reverse_cloud_minus_one, state.position])
        
        reverse_log_densities, reverse_grads = vmap_val_and_grad(reverse_cloud)
        reverse_drifts = jax.vmap(get_drift)(reverse_cloud, reverse_grads)
        
        def compute_reverse_q_back(z_j, z_j_drift):
            return log_q(z_j, candidate_position, z_j_drift)
            
        log_q_y_given_z = jax.vmap(compute_reverse_q_back)(reverse_cloud, reverse_drifts)
        reverse_log_weights = reverse_log_densities + log_q_y_given_z
        
        max_rev_weight = jnp.max(reverse_log_weights)
        stabilized_rev_weights = jnp.exp(reverse_log_weights - max_rev_weight)
        sum_reverse_weights = jnp.sum(stabilized_rev_weights)
        
        # 6. Exact MTM Acceptance Ratio
        log_accept_ratio = (max_log_weight + jnp.log(sum_forward_weights)) - (max_rev_weight + jnp.log(sum_reverse_weights))
        acceptance_rate = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        acceptance_rate = jnp.where(jnp.isnan(acceptance_rate), 0.0, acceptance_rate)
        
        safe_probabilities = jnp.where(
            jnp.isnan(probabilities).any(),
            jnp.ones(num_proposals) / num_proposals,
            probabilities
        )
        
        is_accepted = jax.random.uniform(key_accept) < acceptance_rate
        
        # 7. Resolve State Updates
        next_position = jnp.where(is_accepted, candidate_position, state.position)
        next_log_density = jnp.where(is_accepted, candidate_log_density, state.log_density)
        
        next_state = SlingshotState(position=next_position, log_density=next_log_density)
        info = SlingshotInfo(
            proposal_cloud=proposal_cloud,
            weights=safe_probabilities,
            chosen_index=chosen_index,
            acceptance_rate=acceptance_rate,
            is_accepted=is_accepted
        )
        
        return next_state, info
        
    return one_step

def as_top_level_api(
    logdensity_fn: Callable,
    step_size: float,
    num_proposals: int,
    cholesky: jnp.ndarray = None,
) -> SamplingAlgorithm:
    """User-facing interface factory for the exact Slingshot MP-MCMC sampler."""
    kernel = build_kernel()
    
    return build_sampling_algorithm(
        kernel,
        init,
        logdensity_fn,
        init_args=(),
        kernel_args=(step_size, num_proposals, cholesky),
    )