import jax
import jax.numpy as jnp
from blackjax.util import run_eca
import blackjax.adaptation.ensemble_umclmc as umclmc


def emaus(
    logdensity_fn,
    sample_init,
    transform,
    ndims,
    num_steps1,
    num_steps2,
    num_chains,
    mesh,
    rng_key,
    alpha=1.9,
    save_frac=0.2,
    C=0.1,
    early_stop=True,
    r_end=5e-3,
    diagonal_preconditioning=True,
    integrator_coefficients=None,
    steps_per_sample=15,
    acc_prob=None,
    observables=lambda x: None,
    ensemble_observables=None,
    diagnostics=True,
):
    """
    model: the target density object
    num_steps1: number of steps in the first phase
    num_steps2: number of steps in the second phase
    num_chains: number of chains
    mesh: the mesh object, used for distributing the computation across cpus and nodes
    rng_key: the random key
    alpha: L = sqrt{d}*alpha*variances
    save_frac: the fraction of samples used to estimate the fluctuation in the first phase
    C: constant in stage 1 that determines step size (eq (9) of EMAUS paper)
    early_stop: whether to stop the first phase early
    r_end
    diagonal_preconditioning: whether to use diagonal preconditioning
    integrator_coefficients: the coefficients of the integrator
    steps_per_sample: the number of steps per sample
    acc_prob: the acceptance probability
    observables: the observables (for diagnostic use)
    ensemble_observables:  observable calculated over the ensemble (for diagnostic use)
    diagnostics: whether to return diagnostics
    """

    # observables_for_bias, contract = bias(model)
    key_init, key_umclmc, key_mclmc = jax.random.split(rng_key, 3)

    # initialize the chains
    initial_state = umclmc.initialize(
        key_init, logdensity_fn, sample_init, num_chains, mesh
    )

    # burn-in with the unadjusted method #
    kernel = umclmc.build_kernel(logdensity_fn)
    save_num = (int)(jnp.rint(save_frac * num_steps1))
    adap = umclmc.Adaptation(
        ndims,
        alpha=alpha,
        bias_type=3,
        save_num=save_num,
        C=C,
        power=3.0 / 8.0,
        r_end=r_end,
        observables_for_bias=lambda position: jnp.square(
            transform(jax.flatten_util.ravel_pytree(position)[0])
        ),
    )

    final_state, final_adaptation_state, info1 = run_eca(
        key_umclmc,
        initial_state,
        kernel,
        adap,
        num_steps1,
        num_chains,
        mesh,
        ensemble_observables,
        early_stop=early_stop,
    )