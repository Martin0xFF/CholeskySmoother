from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from montecarlo import MCMC


def generate_velocity(t):
    """Construct simulated trajectory derivative"""
    dx = 0.05 * (np.cos(t) - t * np.sin(t))
    dy = 0.05 * (np.sin(t) + t * np.cos(t))
    dz = 0.05 * np.ones(t.shape)
    return np.hstack([dx[:, None], dy[:, None], dz[:, None]])


def generate_trajectory(t):
    """Construct simulated trajectory that appears as a cork screw."""

    x = 0.05 * t * np.cos(t)
    y = 0.05 * t * np.sin(t)
    z = 0.05 * t
    r = np.hstack([x[:, None], y[:, None], z[:, None]])

    # Axis angle
    dr = generate_velocity(t)
    axis_angle = (
        2 * np.pi * (0.1) * t[:, None] * (dr / np.linalg.norm(dr, axis=1)[:, None])
    )
    return np.hstack([axis_angle, r])


def get_random(ovar, pvar, seed=42, dim=100):
    rng = np.random.default_rng(seed)
    return np.hstack(
        [rng.normal(0, ovar, size=(dim, 3)), rng.normal(0, pvar, size=(dim, 3))]
    )


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


def constructA(dt: float):
    aa = np.array(
        [
            [1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0, 0],
        ]
    )

    p = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt * dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0.5 * dt * dt],
        ]
    )

    av = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    )

    v = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, dt],
        ]
    )

    a = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Constant Process matrix
    return np.vstack([aa, p, av, v, a])


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


def construct_model_martix(op, include_cov=False):
    """Construct the Model Matrix H which expresses the process and observation
    model across K+1 measurements(i.e. entire trajectory).
    """

    # Time delta between samples
    A_o = constructA(op.dt)
    A_inv = np.identity(op.N * (op.K + 1))

    # Populate off diagonal.
    offset = lambda i: op.N * i
    for i in range(op.K):
        A_inv[offset(i + 1) : offset(i + 2), offset(i) : offset(i + 1)] = -A_o

    # Required for state variance estimate.

    if include_cov:
        A = np.linalg.inv(A_inv)
    else:
        A = None

    C_o = np.zeros((op.M, op.N))
    C_o[:, : op.M] = np.identity(op.M)
    C = np.zeros((op.M * (op.K + 1), op.N * (op.K + 1)))

    # Populate diagonal
    for i in range(op.K + 1):
        C[op.M * i : op.M * (i + 1), op.N * i : op.N * (i + 1)] = C_o

    # Return H matrix.
    return (np.vstack([A_inv, C]), A, C)


def construct_noise_matrix(op, include_cov=False):
    # Noise Matrix. First N*(K+1) is related to motion, next M*(K+1) is related
    # to observation.
    N = op.N
    M = op.M
    K = op.K

    W = np.identity((N + M) * (K + 1))
    for i in range(K + 1):
        offset = lambda i: N * (K + 1) + M * i
        # Assign Measured AngleAxis Noise.
        ori_offset = offset(i)
        W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = np.diag(
            np.square(op.mosig) * np.ones((3,))
        )

        # Assign Measured Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = np.diag(
            np.square(op.mpsig) * np.ones((3,))
        )

        if i == 0:
            # initial, treat noise like observation noise since we pick initial
            # state from observation.
            # Assign AngleAxis Noise.
            ori_offset = offset(i)
            W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = np.diag(
                np.square(op.iosig) * np.ones((3,))
            )

            # Assign Position Noise.
            pos_offset = offset(i) + 3
            W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = np.diag(
                np.square(op.ipsig) * np.ones((3,))
            )

            # Assign Angular Velocity Noise.
            vel_offset = offset(i) + 6
            W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
                np.square(op.iovsig) * np.ones((3,))
            )

            # Assign Velocity Noise.
            vel_offset = offset(i) + 9
            W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
                np.square(op.ivsig) * np.ones((3,))
            )

            # Assign Acceleration Noise.
            acc_offset = offset(i) + 12
            W[acc_offset : acc_offset + 3, acc_offset : acc_offset + 3] = np.diag(
                np.square(op.iasig) * np.ones((3,))
            )
            continue

        offset = lambda i: N * i

        # Assign AngleAxis Noise.
        ori_offset = offset(i)
        W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = np.diag(
            np.square(op.osig) * np.ones((3,))
        )

        # Assign Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = np.diag(
            np.square(op.psig) * np.ones((3,))
        )

        # Assign Angular Velocity Noise.
        vel_offset = offset(i) + 6
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
            np.square(op.ovsig) * np.ones((3,))
        )

        # Assign Velocity Noise.
        vel_offset = offset(i) + 9
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
            np.square(op.vsig) * np.ones((3,))
        )

        # Assign Acceleration Noise.
        acc_offset = offset(i) + 12
        W[acc_offset : acc_offset + 3, acc_offset : acc_offset + 3] = np.diag(
            np.square(op.asig) * np.ones((3,))
        )

    # Required for Covariance calculation.
    Q = W[: N * (K + 1), : N * (K + 1)]
    R = W[N * (K + 1) : (N + M) * (K + 1), N * (K + 1) : (N + M) * (K + 1)]
    return W, Q, R


def CholeskySmoother(observations, op, t, include_cov=False):
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

    H, A, C = construct_model_martix(op, include_cov)
    W, Q, R = construct_noise_matrix(op, include_cov)

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


def ConfigureAndRun(state, observations, include_cov=False):
    return CholeskySmoother(
        observations,
        CholeskySmootherOptions(
            N=15,
            M=6,
            K=24,
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
            ipsig=160,
            ivsig=160,
            iasig=160,
            # Models
            psig=state[0],
            vsig=state[1],
            asig=state[2],
            # Observation
            mpsig=state[3],
        ),
        t,
        include_cov,
    )


def SearchForOptimal(t, v, dv, observations):
    def likelihood_fn(x, observations):
        if x[x <= 0.0].any() or x[x >= 5.0].any():
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
        np.array([0.001, 0.005, 0.008, 0.08]),
        1000,
        200,
        12,
    )

    means = np.mean(samples, axis=0)
    for i in range(samples.shape[1]):
        print(f"Final Outlier Less Mean along index {i}: {means[i]}")
    # Include covariance for run with optimal values.
    return ConfigureAndRun(means, observations, include_cov=True)


def ex():
    K = 25
    t = np.linspace(0, 10, K)
    v = generate_trajectory(t)
    dv = generate_velocity(t)
    observations = v + get_random(0.01, 0.04, seed=42, dim=K)
    ConfigureAndRun([1, 1, 1], observations)


if __name__ == "__main__":
    K = 25
    t = np.linspace(0, 10, K)
    v = generate_trajectory(t)
    dv = generate_velocity(t)
    observations = v + get_random(0.01, 0.08, seed=42, dim=K)

    # Time profiling code.
    # import timeit

    # timer = timeit.Timer("ex()", setup="from __main__ import ex")
    # execution_times = timer.repeat(
    #     repeat=3, number=10
    # )  # Run 3 times, each 1000 iterations
    # print(f"Execution times: {execution_times}")
    # print(f"Average execution time: {sum(execution_times)/(3*1000)} seconds")
    # exit(0)

    res, cov = SearchForOptimal(t, v, dv, observations)
    # res, cov = ConfigureAndRun([0.001, 0.04, 0.04], observations, include_cov=True)

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
