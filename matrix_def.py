from typing import Any

import numpy as np
import scipy as sp


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
