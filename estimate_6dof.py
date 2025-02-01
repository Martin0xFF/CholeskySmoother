from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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


def get_random(ovar, pvar, seed=42):
    rng = np.random.default_rng(seed)
    return np.hstack(
        [rng.normal(0, ovar, size=(100, 3)), rng.normal(0, pvar, size=(100, 3))]
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

    osig: float = 1.0
    psig: float = 1.0

    ovsig: float = 1.0
    vsig: float = 1.0
    asig: float = 1.0

    mosig: float = 1.0
    mpsig: float = 1.0


def _construct_model_martix(op):
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
    A = np.linalg.inv(A_inv)

    C_o = np.zeros((op.M, op.N))
    C_o[:, : op.M] = np.identity(op.M)
    C = np.zeros((op.M * (op.K + 1), op.N * (op.K + 1)))

    # Populate diagonal
    for i in range(op.K + 1):
        C[op.M * i : op.M * (i + 1), op.N * i : op.N * (i + 1)] = C_o

    # Return H matrix.
    return (np.vstack([A_inv, C]), A, C)


def _construct_noise_matrix(op):
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
            op.mosig * np.ones((3,))
        )

        # Assign Measured Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = np.diag(
            op.mpsig * np.ones((3,))
        )

        # if i == 0:
        #     # initial, treat noise like observation noise since we pick initial
        #     # state from observation.
        #     continue

        offset = lambda i: N * i

        # Assign AngleAxis Noise.
        ori_offset = offset(i)
        W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = np.diag(
            op.osig * np.ones((3,))
        )

        # Assign Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = np.diag(
            op.psig * np.ones((3,))
        )

        # Assign Angular Velocity Noise.
        vel_offset = offset(i) + 6
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
            op.ovsig * np.ones((3,))
        )

        # Assign Velocity Noise.
        vel_offset = offset(i) + 9
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = np.diag(
            op.vsig * np.ones((3,))
        )

        # Assign Acceleration Noise.
        acc_offset = offset(i) + 12
        W[acc_offset : acc_offset + 3, acc_offset : acc_offset + 3] = np.diag(
            op.asig * np.ones((3,))
        )

    # Required for Covariance calculation.
    Q = W[: N * (K + 1), : N * (K + 1)]
    R = W[N * (K + 1) : (N + M) * (K + 1), N * (K + 1) : (N + M) * (K + 1)]
    return W, Q, R


def CholeskySmoother(observations, op, t):
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

    # Here we assume the initial state and inputs are zero, only populate
    # observations
    offset = lambda i: (K + 1) * N + i * M
    for i in range(K + 1):
        Z[offset(i) : offset(i + 1), 0] = observations[i, :]

    Z[:6, 0] = observations[0, :]
    Z[9:12, 0] = generate_velocity(t)[0, :]

    H, A, C = _construct_model_martix(op)
    W, Q, R = _construct_noise_matrix(op)

    # Covariance Calculation
    P_check = A @ Q @ A.T
    P_hat = np.linalg.inv(np.linalg.inv(P_check) + C.T @ np.linalg.inv(R) @ C)

    # The Key Terms are H,W,Z
    # H - Model Matrix made of Ai and Ci Matricies
    # W - Noise Matrix made of Qi and Ri
    # Z - Xo, Vi, Yi

    W_inv = np.linalg.inv(W)
    X_est = np.linalg.solve(H.T @ W_inv @ H, H.T @ W_inv @ Z)
    return X_est.reshape((K + 1, N), order="C"), P_hat


def trace(index, t, y, res, cov):
    sig = np.sqrt(cov.diagonal()[index::15])
    plt.plot(t, y[:, index])
    plt.plot(t, res[:, index])
    plt.plot(t, res[:, index] + 3 * sig, color="green")
    plt.plot(t, res[:, index] - 3 * sig, color="green")
    plt.show()


if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    v = generate_trajectory(t)
    dv = generate_velocity(t)

    observations = v + get_random(0.01, 0.08, seed=42)
    scale = 1e-3
    options = CholeskySmootherOptions(
        N=15,
        M=6,
        K=99,
        dt=0.1,
        osig=1.0,
        ovsig=1.0,
        psig=scale * 0.1,
        vsig=scale * 0.5,
        asig=scale * 1.0,
        mpsig=scale * 10,
        mosig=1.0,
    )

    res, cov = CholeskySmoother(observations, options, t)

    print(
        f"L2 Norm: {1e3*np.linalg.norm(observations[:,3:] - v[:,3:])/options.K} No Smooth Dist mm per frame"
    )
    print(
        f"L2 Norm: {1e3*np.linalg.norm(res[:, 3:6] - v[:,3:])/options.K} With Smooth Dist mm per frame"
    )
    print(
        f"L2 Norm: {1e3*np.linalg.norm(res[:, 9:12] - dv[:,:])/options.K} With Smooth Vel Error mm/s per frame"
    )
    print(
        f"L2 Norm: {np.linalg.norm(observations[:,:3] - v[:,:3])/options.K} No Smooth Ori rad per frame"
    )
    print(
        f"L2 Norm: {np.linalg.norm(res[:, :3] - v[:,:3])/options.K} With Smooth Ori rad per frame"
    )

    trace(3, t, v, res, cov)
    trace(4, t, v, res, cov)
    trace(5, t, v, res, cov)

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
