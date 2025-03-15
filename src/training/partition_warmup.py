"""Multiple warmup implementations for different samplers."""
from functools import partial
from pathlib import Path
from typing import Callable

import blackjax
import blackjax.mcmc as mcmc
import jax
import jax.experimental
import jax.numpy as jnp
from blackjax.adaptation.base import AdaptationResults
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState
from blackjax.adaptation.window_adaptation import base, build_schedule
from blackjax.base import (
    AdaptationAlgorithm,
    ArrayLikeTree,
    PRNGKey,
)
from blackjax.diagnostics import effective_sample_size
from blackjax.util import pytree_size, streaming_average_update
from jax.flatten_util import ravel_pytree

from src.training.callbacks import progress_bar_scan, save_position


# TAKEN FROM BLACKJAX
def custom_window_adaptation(
    algorithm,
    logdensity_fn: Callable,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.80,
    progress_bar: bool = False,
    saving_path: Path | None = None,
    **extra_parameters,
) -> AdaptationAlgorithm:  # noqa
    """Adapt the value of the inverse mass matrix and step size parameters of
    algorithms in the HMC fmaily.

    Algorithms in the HMC family on a euclidean manifold depend on the value of
    at least two parameters: the step size, related to the trajectory
    integrator, and the mass matrix, linked to the euclidean metric.

    Good tuning is very important, especially for algorithms like NUTS which can
    be extremely inefficient with the wrong parameter values. This function
    provides a general-purpose algorithm to tune the values of these parameters.
    Originally based on Stan's window adaptation, the algorithm has evolved to
    improve performance and quality.

    Parameters
    ----------
    algorithm
        The algorithm whose parameters are being tuned.
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    is_mass_matrix_diagonal
        Whether we should adapt a diagonal mass matrix.
    initial_step_size
        The initial step size used in the algorithm.
    target_acceptance_rate
        The acceptance rate that we target during step size adaptation.
    progress_bar
        Whether we should display a progress bar.
    saving_path
        The path to save the warmup positions for each state.
    **extra_parameters
        The extra parameters to pass to the algorithm, e.g. the number of
        integration steps for HMC.

    Returns
    -------
    A function that runs the adaptation and returns an `AdaptationResult` object.

    """

    mcmc_kernel = algorithm.build_kernel()

    adapt_init, adapt_step, adapt_final = base(
        is_mass_matrix_diagonal,
        target_acceptance_rate=target_acceptance_rate,
    )


    def run(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        device_id: jnp.ndarray,
        hidden_layers: dict,
        num_steps: int = 1000,
        n_devices: int = 1,
    ):
        partition_logdensity_fn = partial(logdensity_fn, hidden_layers=hidden_layers)
        init_state = algorithm.init(position, partition_logdensity_fn)
        init_adaptation_state = adapt_init(position, initial_step_size)

        def one_step(carry, xs):
            n, rng_key, adaptation_stage, device_id = xs
            state, adaptation_state = carry

            new_state, info = mcmc_kernel(
                rng_key,
                state,
                partition_logdensity_fn,
                adaptation_state.step_size,
                adaptation_state.inverse_mass_matrix,
                **extra_parameters,
            )
            new_adaptation_state = adapt_step(
                adaptation_state,
                adaptation_stage,
                new_state.position,
                info.acceptance_rate,
            )
            if saving_path:
                jax.experimental.io_callback(
                    partial(save_position, base=saving_path),
                    result_shape_dtypes=state.position,
                    position=state.position,
                    idx=device_id,
                    n=n,
                )

            return (new_state, new_adaptation_state), jnp.nan
    
        if progress_bar:
            one_step_ = jax.jit(
                progress_bar_scan(
                    n_steps=num_steps * n_devices, name='Window Adaptation'
                )(one_step)
            )
        else:
            one_step_ = jax.jit(one_step)

        keys = jax.random.split(rng_key, num_steps)
        schedule = build_schedule(num_steps)
        last_state, _ = jax.lax.scan(
            one_step_,
            (init_state, init_adaptation_state),
            (jnp.arange(num_steps), keys, schedule, device_id),
        )
        last_chain_state, last_warmup_state, *_ = last_state

        step_size, inverse_mass_matrix = adapt_final(last_warmup_state)
        parameters = {
            'step_size': step_size,
            'inverse_mass_matrix': inverse_mass_matrix,
            **extra_parameters,
        }

        return AdaptationResults(last_chain_state, parameters)

    return AdaptationAlgorithm(run)


# This routine is taken from blackjax.mclmc and then modified to fit
# the needs of the BNN use case.
def mclmc_find_L_and_step_size(
    mclmc_kernel,
    state: MCLMCAdaptationState,
    rng_key: jax.random.PRNGKey,
    tune1_steps: int = 100,
    tune2_steps: int = 100,
    tune3_steps: int = 100,
    step_size_init: float = 0.005,
    desired_energy_var_start: float = 5e-4,
    desired_energy_var_end: float = 5e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: int = 150,
    diagonal_preconditioning: bool = False,
):
    """
    Find the optimal value of the parameters for the MCLMC algorithm.

    Parameters
    ----------
    mclmc_kernel
        The kernel function used for the MCMC algorithm.
    state
        The initial state of the MCMC algorithm.
    rng_key
        The random number generator key.
    tune1_steps
        The number of steps for the first step of the adaptation.
    tune2_steps
        The number of steps for the second step of the adaptation.
    tune3_steps
        The number of steps for the third step of the adaptation.
    step_size_init
        The initial step size for the MCMC algorithm.
    desired_energy_var_start
        The desired energy variance for the MCMC algorithm.
    desired_energy_var_end
        The desired energy variance for the MCMC algorithm at the end of the
        linear decay schedule. If the value is the same as the start value, no
        decay is applied.
    trust_in_estimate
        The trust in the estimate of optimal stepsize.
    num_effective_samples
        The number of effective samples for the MCMC algorithm.

    Returns
    -------
    A tuple containing the final state of the MCMC algorithm and the final
    hyperparameters.
    """
    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(
        L=jnp.maximum(jnp.sqrt(dim), 15.0),
        step_size=step_size_init,
        sqrt_diag_cov=jnp.ones((dim,)),
    )
    part1_key, part2_key = jax.random.split(rng_key, 2)
    state, params = make_L_step_size_adaptation(
        kernel=mclmc_kernel,
        dim=dim,
        tune1_steps=tune1_steps,
        tune2_steps=tune2_steps,
        desired_energy_var_start=desired_energy_var_start,
        desired_energy_var_end=desired_energy_var_end,
        trust_in_estimate=trust_in_estimate,
        num_effective_samples=num_effective_samples,
        diagonal_preconditioning=diagonal_preconditioning,
    )(state, params, part1_key)

    if tune3_steps != 0:
        state, params = make_adaptation_L(
            mclmc_kernel(params.sqrt_diag_cov), Lfactor=0.4
        )(state, params, tune3_steps, part2_key)

    return state, params


def make_L_step_size_adaptation(
    kernel,
    dim: int,
    tune1_steps: int,
    tune2_steps: int,
    diagonal_preconditioning: bool,
    desired_energy_var_start: float,
    desired_energy_var_end: float,
    trust_in_estimate: float,
    num_effective_samples: int,
):
    """
    Adapt the stepsize and L of the MCLMC kernel.

    Designed for the unadjusted MCLMC
    """
    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    def get_desired_energy_var_linear(step: int) -> float:
        """Linearly decrease desired_energy_var from start to end."""
        total_steps = tune1_steps + tune2_steps + 1
        progress = jax.numpy.minimum(step / total_steps, 1.0)
        return (
            desired_energy_var_start
            - (desired_energy_var_start - desired_energy_var_end) * progress
        )

    def get_desired_energy_var_exp(step: int) -> float:
        """Exponentially decrease desired_energy_var from start to end."""
        total_steps = tune1_steps + tune2_steps + 1
        tau = total_steps / 4
        return desired_energy_var_start * jax.numpy.exp(
            -step / tau
        ) + desired_energy_var_end * (1 - jax.numpy.exp(-step / tau))

    if desired_energy_var_start > 2.0:
        get_desired_energy_var = get_desired_energy_var_exp
    else:
        get_desired_energy_var = get_desired_energy_var_linear

    def predictor(
        previous_state,
        params,
        adaptive_state: MCLMCAdaptationState,
        rng_key: jax.random.PRNGKey,
        step_number: int,
    ):
        """
        Do one step with the dynamics and updates the prediction for the opt. stepsize.

        Designed for the unadjusted MCHMC
        """
        time, x_average, step_size_max = adaptive_state

        # dynamics
        next_state, info = kernel(params.sqrt_diag_cov)(
            rng_key=rng_key,
            state=previous_state,
            L=params.L,
            step_size=params.step_size,
        )
        # step updating
        success, state, step_size_max, energy_change = handle_nans(
            previous_state,
            next_state,
            params.step_size,
            step_size_max,
            info.energy_change,
        )

        # Warning: var = 0 if there were nans, but we will give it a very small weight
        desired_energy_var = get_desired_energy_var(step_number)
        xi = (
            jnp.square(energy_change) / (dim * desired_energy_var)
        ) + 1e-8  # 1e-8 is added to avoid divergences in log xi
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )  # the weight reduces the impact of stepsizes which are much larger
        # or much smaller than the desired one.

        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(
            x_average / time, -1.0 / 6.0
        )  # We use the Var[E] = O(eps^6) relation here.
        step_size = (step_size < step_size_max) * step_size + (
            step_size > step_size_max
        ) * step_size_max  # if the proposed stepsize is above the stepsize
        # where we have seen divergences
        params_new = params._replace(step_size=step_size)

        adaptive_state = (time, x_average, step_size_max)

        return state, params_new, adaptive_state, success

    def step(iteration_state, weight_key_step):
        """Do one step and update the estimates of the post. & step size."""
        mask, rng_key, step_number = weight_key_step
        state, params, adaptive_state, streaming_avg = iteration_state

        state, params, adaptive_state, success = predictor(
            state,
            params,
            adaptive_state,
            rng_key,
            step_number,
        )

        x = ravel_pytree(state.position)[0]
        # update the running average of x, x^2
        streaming_avg = streaming_average_update(
            current_value=jnp.array([x, jnp.square(x)]),
            previous_weight_and_average=streaming_avg,
            weight=(1 - mask) * success * params.step_size,
            zero_prevention=mask,
        )

        return (state, params, adaptive_state, streaming_avg), None

    def run_steps(xs, state, params):
        step_number = jnp.arange(len(xs[0]))
        return jax.lax.scan(
            step,
            init=(
                state,
                params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=(xs[0], xs[1], step_number),
        )[0]

    def L_step_size_adaptation(state, params, rng_key):
        """Initialize adaptation of the step size and L."""
        L_step_size_adaptation_keys = jax.random.split(
            rng_key, tune1_steps + tune2_steps + 1
        )
        L_step_size_adaptation_keys, final_key = (
            L_step_size_adaptation_keys[:-1],
            L_step_size_adaptation_keys[-1],
        )

        # we use the last num_steps2 to compute the diagonal preconditioner
        mask = 1 - jnp.concatenate((jnp.zeros(tune1_steps), jnp.ones(tune2_steps)))

        # run the steps
        state, params, _, (_, average) = run_steps(
            xs=(mask, L_step_size_adaptation_keys), state=state, params=params
        )

        L = params.L
        # determine L
        sqrt_diag_cov = params.sqrt_diag_cov
        if tune2_steps != 0.0:
            x_average, x_squared_average = average[0], average[1]
            variances = x_squared_average - jnp.square(x_average)
            # jax.debug.print('Average variances: {x}', x=jnp.mean(variances))
            L = jnp.sqrt(jnp.sum(variances))
            if diagonal_preconditioning:
                sqrt_diag_cov = jnp.sqrt(variances)
                params = params._replace(sqrt_diag_cov=sqrt_diag_cov)
                L = jnp.sqrt(dim)

                # readjust the stepsize
                steps = tune2_steps // 3  # we do some small number of steps
                keys = jax.random.split(final_key, steps)
                state, params, _, (_, average) = run_steps(
                    xs=(jnp.ones(steps), keys), state=state, params=params
                )

        return state, MCLMCAdaptationState(L, params.step_size, sqrt_diag_cov)

    return L_step_size_adaptation


def make_adaptation_L(kernel, Lfactor):
    """
    Determine L by the autocorrelations .

    (around 10 effective samples are needed for this to be accurate)
    """
    Lfactor = Lfactor

    def adaptation_L(
        state,
        params,
        num_steps: int,
        key: jax.random.PRNGKey,
        fft_params_limit: int = 2000,
        fft_samples_limit: int = 10000,
    ):
        adaptation_L_keys = jax.random.split(key, num_steps)

        def step(state, key):
            next_state, _ = kernel(
                rng_key=key,
                state=state,
                L=params.L,
                step_size=params.step_size,
            )

            return next_state, next_state.position

        state, samples = jax.lax.scan(
            f=step,
            init=state,
            xs=adaptation_L_keys,
        )

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
        # Limit the number of samples and parameters to compute the FFT
        if flat_samples.shape[1] > fft_params_limit:
            flat_samples = flat_samples[
                :,
                jax.random.permutation(key, jnp.arange(flat_samples.shape[1]))[
                    :fft_params_limit
                ],
            ]
        if flat_samples.shape[0] > fft_samples_limit:
            # here not random but equally spaced (thinning)
            flat_samples = flat_samples[
                jnp.linspace(0, flat_samples.shape[0] - 1, fft_samples_limit).astype(
                    jnp.int32
                )
            ]
        ess = effective_sample_size(flat_samples[None, ...])
        # jax.debug.print("ESS: {x}", x=ess)

        return state, params._replace(
            L=Lfactor * params.step_size * jnp.mean(num_steps / ess)
        )

    return adaptation_L


def handle_nans(previous_state, next_state, step_size, step_size_max, kinetic_change):
    """
    Handle nans in the state.

    if there are nans, let's reduce the stepsize, and not update the state. The
    function returns the old state in this case.
    """
    reduced_step_size = 0.8
    p, _ = ravel_pytree(next_state.position)
    nonans = jnp.all(jnp.isfinite(p))
    state, step_size, kinetic_change = jax.tree_util.tree_map(
        lambda new, old: jax.lax.select(nonans, jnp.nan_to_num(new), old),
        (next_state, step_size_max, kinetic_change),
        (previous_state, step_size * reduced_step_size, 0.0),
    )
    return nonans, state, step_size, kinetic_change


def custom_mclmc_warmup(
    logdensity_fn: Callable,
    diagonal_preconditioning: bool = True,
    desired_energy_var_start: float = 5e-4,
    desired_energy_var_end: float = 5e-4,
    trust_in_estimate: float = 1.5,
    num_effective_samples: int = 100,
    step_size_init: float = 0.005,
    # TODO add saving_path: Path | None = None,
) -> AdaptationAlgorithm:
    """Warmup the initial state using MCLMC.

    Parameters
    ----------
    logdensity_fn
        The log density probability density function from which we wish to
        sample.
    diagonal_preconditioning
        Whether we should use diagonal preconditioning.
    desired_energy_var_start
        The desired energy variance at the start of the linear decay schedule.
    desired_energy_var_end
        The desired energy variance at the end of the linear decay schedule. If the
        value is the same as the start value, no decay is applied.
    trust_in_estimate
        The trust in the estimate.
    num_effective_samples
        The number of effective samples.

    Returns
    -------
    AdaptationResults
        The current state of the chain and the parameters for the sampler.
    """
    diagonal_preconditioning = diagonal_preconditioning


    def run(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        num_steps: int = 1000,
        hidden_layers: dict = None,
    ):
        partition_logdensity_fn = partial(logdensity_fn, hidden_layers=hidden_layers)
        """Run the MCLMC warmup."""
        initial_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=partition_logdensity_fn, rng_key=rng_key
        )
        # TODO: Parameterize the phase ratio
        phase_ratio = (0.8, 0.1, 0.1)

        def kernel(sqrt_diag_cov):
            """Build the MCLMC kernel."""
            return mcmc.mclmc.build_kernel(
                logdensity_fn=partition_logdensity_fn,
                # TODO integrator should be parameterized in config
                # TODO blackjax build_kernel should be changed to accept the integrator
                # integrator=mcmc.integrators.velocity_verlet,
                integrator=mcmc.integrators.isokinetic_mclachlan,
                sqrt_diag_cov=sqrt_diag_cov,
            )
        
        # find values for L and step size
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            state=initial_state,
            rng_key=rng_key,
            diagonal_preconditioning=diagonal_preconditioning,
            step_size_init=step_size_init,
            tune1_steps=int(num_steps * phase_ratio[0]),
            tune2_steps=int(num_steps * phase_ratio[1]),
            tune3_steps=int(num_steps * phase_ratio[2]),
            desired_energy_var_start=desired_energy_var_start,
            desired_energy_var_end=desired_energy_var_end,
            trust_in_estimate=trust_in_estimate,
            num_effective_samples=num_effective_samples,
        )
        return AdaptationResults(
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        )

    return AdaptationAlgorithm(run)
