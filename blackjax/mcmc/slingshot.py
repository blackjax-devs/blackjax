import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

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

def init(position: jnp.ndarray, logdensity_fn: Callable) -> SlingshotState:
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

def dual_averaging_step(
    adapt_state: SlingshotAdaptState,
    acceptance_rate: float,
    position: jnp.ndarray,
    target_rate: float = 0.65,
    gamma: float = 0.05,
    t0: int = 10,
    kappa: float = 0.75
) -> SlingshotAdaptState:
    """Update step size logs dynamically using Nesterov dual averaging and dense covariance."""
    t = adapt_state.t + 1
    alpha = target_rate - acceptance_rate
    h_bar = (1.0 - 1.0 / (t + t0)) * adapt_state.h_bar + (1.0 / (t + t0)) * alpha
    
    log_step_size = - (jnp.sqrt(t) / gamma) * h_bar
    eta = t ** (-kappa)
    log_step_size_bar = (1.0 - eta) * adapt_state.log_step_size_bar + eta * log_step_size
    
    # Welford's algorithm for dense covariance
    delta = position - adapt_state.mean
    mean = adapt_state.mean + delta / t
    delta2 = position - mean
    m2 = adapt_state.m2 + jnp.outer(delta, delta2)
    
    # Recalculate Cholesky occasionally or every step
    # Regularize with identity to ensure positive definiteness
    cov = m2 / jnp.maximum(1, t - 1)
    cov = cov + 1e-5 * jnp.eye(cov.shape[0])
    cholesky = jax.scipy.linalg.cholesky(cov, lower=True)
    
    return SlingshotAdaptState(
        log_step_size=log_step_size,
        log_step_size_bar=log_step_size_bar,
        h_bar=h_bar,
        t=t,
        mean=mean,
        m2=m2,
        cholesky=cholesky
    )

def kernel() -> Callable:
    """Build the functional transition kernel for the Slingshot sampler.

    Evaluates both forward and reverse clouds via jax.vmap to run an exact
    Metropolis-Hastings validation step matching finite-proposal conditions.
    """
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
        vmapped_logdensity = jax.vmap(logdensity_fn)
        
        if cholesky is None:
            cholesky = jnp.eye(dim)
            
        def compute_mahalanobis(points, center, L):
            diffs = points - center
            scaled_diffs = jax.scipy.linalg.solve_triangular(L, diffs.T, lower=True).T
            return jnp.linalg.norm(scaled_diffs, axis=-1)
        
        # 1. Forward Pass: Generate and evaluate proposal cloud
        raw_noise = jax.random.normal(key_cloud, shape=(num_proposals, dim))
        noise = jnp.dot(raw_noise, cholesky.T)
        proposal_cloud = state.position + noise * step_size
        cloud_log_densities = vmapped_logdensity(proposal_cloud)
        
        distances = compute_mahalanobis(proposal_cloud, state.position, cholesky)
        log_weights = cloud_log_densities + jnp.log(distances + 1e-8)
        
        max_log_weight = jnp.max(log_weights)
        stabilized_weights = jnp.exp(log_weights - max_log_weight)
        sum_forward_weights = jnp.sum(stabilized_weights)
        probabilities = stabilized_weights / sum_forward_weights
        
        # 2. Extract Candidate Proposal
        chosen_index = jax.random.choice(key_select, num_proposals, p=probabilities)
        candidate_position = proposal_cloud[chosen_index]
        candidate_log_density = cloud_log_densities[chosen_index]
        
        # 3. Reverse Pass: Generate validation cloud around candidate position
        raw_reverse_noise = jax.random.normal(key_reverse, shape=(num_proposals - 1, dim))
        reverse_noise = jnp.dot(raw_reverse_noise, cholesky.T)
        reverse_cloud_minus_one = candidate_position + reverse_noise * step_size
        reverse_cloud = jnp.vstack([reverse_cloud_minus_one, state.position]) # Append original state
        
        reverse_log_densities = vmapped_logdensity(reverse_cloud)
        reverse_distances = compute_mahalanobis(reverse_cloud, candidate_position, cholesky)
        reverse_log_weights = reverse_log_densities + jnp.log(reverse_distances + 1e-8)
        
        max_rev_weight = jnp.max(reverse_log_weights)
        stabilized_rev_weights = jnp.exp(reverse_log_weights - max_rev_weight)
        sum_reverse_weights = jnp.sum(stabilized_rev_weights)
        
        # 4. Exact Finite-P Metropolis Acceptance Correction
        log_accept_ratio = (max_log_weight + jnp.log(sum_forward_weights)) - (max_rev_weight + jnp.log(sum_reverse_weights))
        acceptance_rate = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        
        is_accepted = jax.random.uniform(key_accept) < acceptance_rate
        
        # 5. Resolve State Updates
        next_position = jnp.where(is_accepted, candidate_position, state.position)
        next_log_density = jnp.where(is_accepted, candidate_log_density, state.log_density)
        
        next_state = SlingshotState(position=next_position, log_density=next_log_density)
        info = SlingshotInfo(
            proposal_cloud=proposal_cloud,
            weights=probabilities,
            chosen_index=chosen_index,
            acceptance_rate=acceptance_rate,
            is_accepted=is_accepted
        )
        
        return next_state, info
        
    return one_step