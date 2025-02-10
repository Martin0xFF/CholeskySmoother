from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import matrix_def as md
from montecarlo import MCMC


def plot(lst_arr, lst_other=[], zlim=[]):
    fig = plt.figure(figsize=plt.figaspect(9 / 16))
    col = 2 if lst_other else 1

    ax = fig.add_subplot(1, col, 1, projection="3d")
    if zlim:
        ax.set_zlim(zlim[0], zlim[1])
    for arr in lst_arr:
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2])

    if lst_other:
        ax2 = fig.add_subplot(
            1, col, 2, projection="3d", sharex=ax, sharey=ax, sharez=ax, shareview=ax
        )
        for arr in lst_other:
            ax2.scatter(arr[:, 0], arr[:, 1], arr[:, 2])
    plt.show()


@dataclass
class CholeskySmootherOptions:
    N: int = 15
    M: int = 6
    K: int = 99

    dt: float = 0.1

    # Note this is standard deviation and not variance.
    osig: float = 1.0
    psig: float = 1.0

    ovsig: float = 1.0
    vsig: float = 1.0
    asig: float = 1.0

    iosig: float = 1.0
    ipsig: float = 1.0

    iovsig: float = 1.0
    ivsig: float = 1.0
    iasig: float = 1.0

    mosig: float = 1.0
    mpsig: float = 1.0


def CholeskySmoother(observations, op, t, include_cov=False, sparse=False):
    """Given 6DoF Observations structure as convex problem over trajectory and
    solve with cholesky decomposition.

    Assume the state xi we are tracking is:
    - angleaxis 3
    - position 3
    - angular velocity 3
    - velocity 3
    - acceleration 3
    Total: 15

    Here I will assume that:
    - Ak = A0 - process model
    - Qk = Q0 - process noise
    - Ck = C0 - observation model
    - Rk = R0 - observation noise
    """

    # Number of internal states.
    N = op.N
    # Number of observed states.
    M = op.M
    # K - max step index in the trajectory
    K = op.K

    # Construct Initial state, Inputs and Observations vector.
    Z = np.zeros(((N + M) * (K + 1), 1))

    # Here we assume the inputs are zero, only populate
    # observations
    offset = lambda i: (K + 1) * N + i * M
    for i in range(K + 1):
        Z[offset(i) : offset(i + 1), 0] = observations[i, :]

    # Use initial observation for initial position.
    Z[:6, 0] = observations[0, :]
    # No information about derivatives so just assume zero.

    H, A, C = md.construct_model_martix(op, include_cov)
    W, Q, R = md.construct_noise_matrix(op, include_cov)

    if sparse:
        sH, sA, sC = md.construct_sparse_model_martix(op, include_cov)
        sW, sQ, sR = md.construct_sparse_noise_matrix(op, include_cov)

        sW_inv = np.diag(1.0 / sW.diagonal())
        X_est = sp.sparse.linalg.spsolve(sH.T @ sW_inv @ sH, sH.T @ sW_inv @ Z)

        if include_cov:
            sA = sp.sparse.csr_matrix(A)
            sC = sp.sparse.csr_matrix(C)

            sQ = sp.sparse.csr_matrix(Q)
            sR = sp.sparse.csr_matrix(R)

            sP_check = sA @ sQ @ sA.T
            sP_hat = sp.sparse.linalg.inv(
                sp.sparse.linalg.inv(sP_check) + sC.T @ sp.sparse.linalg.inv(sR) @ sC
            )
        else:
            sP_hat = None
        return X_est.reshape((K + 1, N), order="C"), sP_hat

    # Covariance Calculation
    if include_cov:
        P_check = A @ Q @ A.T
        P_hat = np.linalg.inv(np.linalg.inv(P_check) + C.T @ np.linalg.inv(R) @ C)
    else:
        P_hat = None

    # The Key Terms are H,W,Z
    # H - Model Matrix made of Ai and Ci Matricies
    # W - Noise Matrix made of Qi and Ri
    # Z - Xo, Vi, Yi

    W_inv = np.diag(1.0 / W.diagonal())
    # W_inv = np.linalg.inv(W)
    X_est = np.linalg.solve(H.T @ W_inv @ H, H.T @ W_inv @ Z)
    return X_est.reshape((K + 1, N), order="C"), P_hat


def trace(index, t, y, res, cov, error=False):
    # TODO: Compute the Eigen values of the covariance matrix then plot those
    # instead of just diagonal.
    sig = np.sqrt(cov.diagonal()[index::15])
    err = y[:, index] - res[:, index]
    if error:
        plt.plot(t, err)
        plt.plot(t, err + 3 * sig, color="green", linestyle="dashed")
        plt.plot(t, err - 3 * sig, color="green", linestyle="dashed")
    else:
        plt.plot(t, y[:, index])
        plt.plot(t, res[:, index])
        plt.plot(t, res[:, index] + 3 * sig, color="green", linestyle="dashed")
        plt.plot(t, res[:, index] - 3 * sig, color="green", linestyle="dashed")
    plt.show()


def ConfigureAndRun(state, observations, include_cov=False, sparse=False):
    return CholeskySmoother(
        observations,
        CholeskySmootherOptions(
            N=15,
            M=6,
            K=39,
            dt=0.1,
            # Orientation
            # Initial
            iosig=1.0,
            iovsig=1.0,
            # Model
            osig=1.0,
            ovsig=1.0,
            # Observation
            mosig=1.0,
            # Position
            # initial
            ipsig=state[4],
            ivsig=state[5],
            iasig=state[6],
            # Models
            psig=state[0],
            vsig=state[1],
            asig=state[2],
            # Observation
            mpsig=state[3],
        ),
        t,
        include_cov,
        sparse,
    )


def SearchForOptimal(t, v, dv, observations):
    def likelihood_fn(x, observations):
        if x[x <= 0.0].any() or x[x >= 10.0].any():
            return -np.inf
        # Compute Position Norm as likelihood.
        # TODO: Determine why scaling these terms by some large factor helps.
        res, _cov = ConfigureAndRun(x, observations)
        pos_term = -1e6 * np.sum(
            np.square(np.linalg.norm(res[:, 3:6] - v[:, 3:], axis=1))
        )
        vel_term = -1e3 * np.sum(
            np.square(np.linalg.norm(res[:, 9:12] - dv[:, :], axis=1))
        )
        return pos_term + vel_term

    samples = MCMC(
        likelihood_fn,
        [observations],
        np.array([0.001, 0.005, 0.008, 0.08, 1.0, 1.0, 1.0]),
        500,
        200,
        14,
    )

    means = np.mean(samples, axis=0)
    for i in range(samples.shape[1]):
        print(f"Final Outlier Less Mean along index {i}: {means[i]}")
    # Include covariance for run with optimal values.
    return ConfigureAndRun(means, observations, include_cov=True)


def ex(sparse=False):
    K = 40
    t = np.linspace(0, 10, K)
    v = md.generate_trajectory(t)
    dv = md.generate_velocity(t)
    observations = v + md.get_random(0.01, 0.04, seed=42, dim=K)
    ConfigureAndRun(
        [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        observations,
        sparse=sparse,
    )


if __name__ == "__main__":
    import argparse as ap

    K = 40
    t = np.linspace(0, 10, K)
    v = md.generate_trajectory(t)
    dv = md.generate_velocity(t)
    observations = v + md.get_random(0.01, 0.08, seed=42, dim=K)

    parser = ap.ArgumentParser()
    parser.add_argument(
        "--profile",
        action=ap.BooleanOptionalAction,
    )
    parser.add_argument(
        "--search",
        action=ap.BooleanOptionalAction,
    )
    parser.add_argument(
        "--sparse",
        action=ap.BooleanOptionalAction,
    )
    args = parser.parse_args()

    if args.profile:
        # Time profiling code.
        import timeit

        timer = timeit.Timer(
            "ex(sparse=sparse)",
            setup=f"from __main__ import ex; sparse = {args.sparse}",
        )
        execution_times = timer.repeat(
            repeat=3, number=100
        )  # Run 3 times, each 1000 iterations
        print(f"Execution times: {execution_times}")
        print(f"Average execution time: {sum(execution_times)/(3*1000)} seconds")
        exit(0)

    res, cov = (
        SearchForOptimal(t, v, dv, observations)
        if args.search
        else ConfigureAndRun(
            [0.001, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
            observations,
            include_cov=True,
        )
    )

    print(
        f"L2 Norm: {1e3*np.linalg.norm(observations[:,3:] - v[:,3:])/K} No Smooth Dist mm per frame"
    )
    print(
        f"L2 Norm: {1e3*np.linalg.norm(res[:, 3:6] - v[:,3:])/K} With Smooth Dist mm per frame"
    )
    print(
        f"L2 Norm: {1e3*np.linalg.norm(res[:, 9:12] - dv[:,:])/K} With Smooth Vel Error mm/s per frame"
    )
    print(
        f"L2 Norm: {np.linalg.norm(observations[:,:3] - v[:,:3])/K} No Smooth Ori rad per frame"
    )
    print(
        f"L2 Norm: {np.linalg.norm(res[:, :3] - v[:,:3])/K} With Smooth Ori rad per frame"
    )

    trace(3, t, v, res, cov, True)
    trace(4, t, v, res, cov, True)
    trace(5, t, v, res, cov, True)

    # # Plot positions
    plot(
        [v[:, 3:], observations[:, 3:]],
        [v[:, 3:], res[:, 3:]],
    )

    # # Plot velocities
    plot([dv[:, :], res[:, 9:]], zlim=[0.04, 0.06])

    # # Plot Orientations
    # plot(
    #     [v[:, :], observations[:, :]],
    #     [v[:, :], res[:, :]],
    # )
