import emcee
import matplotlib.pyplot as plt
import numpy as np


def generate_multivariate_gaussian(ndim):
    # Here we attempt to learn the following multivariate Gaussian.
    np.random.seed(42)
    means = np.random.rand(ndim)
    print(means)
    cov = 0.5 - np.random.rand(ndim**2).reshape((ndim, ndim))
    cov = np.triu(cov)
    cov += cov.T - np.diag(cov.diagonal())
    cov = np.dot(cov, cov)
    return means, cov


def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))


def MCMC(
    likelihood_fn,
    data,
    initial_values,
    steps=10000,
    burn_in=100,
    nwalkers=32,
    start_var=0.00001,
):
    """Attempts to perform a probabilistic optimization of the variance
    parameters associated with the Cholesky Smoother.

    data - corresponds to the args passed to the likelihood_fn, typically the
    observations
    """
    rng = np.random.default_rng(42)
    ndim = initial_values.shape[0]

    # Configure sampler.
    p0 = initial_values + rng.normal(0, start_var, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        likelihood_fn,
        args=data,
        moves=[
            (emcee.moves.DEMove(), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ],
    )

    # Burn in 100 steps, this number is from the example, likely should tune
    # this.
    state = sampler.run_mcmc(p0, burn_in, progress=True)
    sampler.reset()

    # Perform optimization.
    for k in range(steps // 100):
        state = sampler.run_mcmc(state, 100, progress=True)
        # Samples represent distributions learned from data for target parameters.
        samples = sampler.get_chain(flat=True)
        vals = []
        for i in range(ndim):
            std = np.std(samples[:, i])
            val = np.mean(samples[:, i])
            vals.append(val)
            print(f"Mean along index {i}: {val}, {std}")
        print(f"Loglikeihood: {likelihood_fn(np.array(vals),*data)}")

    fig, axs = plt.subplots(ndim, 1)
    for i in range(ndim):
        axs[i].hist(samples[:, i], 1000, color="k", histtype="step")
    plt.show()
    return samples


if __name__ == "__main__":
    MCMC(log_prob, generate_multivariate_gaussian(3), np.random.rand(3))
