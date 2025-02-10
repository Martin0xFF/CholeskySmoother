from functools import lru_cache
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


@lru_cache
def constructStepA(dt: float):
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


@lru_cache
def constructStepC(M, N):
    C_o = np.zeros((M, N))
    C_o[:, :M] = np.identity(M)
    return C_o


@lru_cache
def constructInvA(dt, N, K):
    # Time delta between samples
    A_o = constructStepA(dt)
    A_inv = np.identity(N * (K + 1))

    # Plate off diagonal.
    offset = lambda i: N * i
    for i in range(K):
        A_inv[offset(i + 1) : offset(i + 2), offset(i) : offset(i + 1)] = -A_o
    return A_inv


def construct_model_martix(op, include_cov=False):
    """Construct the Model Matrix H which expresses the process and observation
    model across K+1 measurements(i.e. entire trajectory).
    """

    A_inv = constructInvA(op.dt, op.N, op.K)
    # Required for state variance estimate.

    if include_cov:
        A = np.linalg.inv(A_inv)
    else:
        A = None

    C_o = constructStepC(op.M, op.N)
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

    # Observation values.
    MO_mat = np.diag(np.square(op.mosig) * np.ones((3,)))
    MP_mat = np.diag(np.square(op.mpsig) * np.ones((3,)))

    # Initial Values.
    IO_mat = np.diag(np.square(op.iosig) * np.ones((3,)))
    IP_mat = np.diag(np.square(op.ipsig) * np.ones((3,)))
    IOV_mat = np.diag(np.square(op.iovsig) * np.ones((3,)))
    IV_mat = np.diag(np.square(op.ivsig) * np.ones((3,)))
    IA_mat = np.diag(np.square(op.iasig) * np.ones((3,)))

    # Process Values.
    O_mat = np.diag(np.square(op.osig) * np.ones((3,)))
    P_mat = np.diag(np.square(op.psig) * np.ones((3,)))
    OV_mat = np.diag(np.square(op.ovsig) * np.ones((3,)))
    V_mat = np.diag(np.square(op.vsig) * np.ones((3,)))
    A_mat = np.diag(np.square(op.asig) * np.ones((3,)))

    for i in range(K + 1):
        offset = lambda i: N * (K + 1) + M * i
        # Assign Measured AngleAxis Noise.
        ori_offset = offset(i)
        W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = MO_mat
        # Assign Measured Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = MP_mat
        if i == 0:
            # initial, treat noise like observation noise since we pick initial
            # state from observation.
            # Assign AngleAxis Noise.
            ori_offset = offset(i)
            W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = IO_mat

            # Assign Position Noise.
            pos_offset = offset(i) + 3
            W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = IP_mat

            # Assign Angular Velocity Noise.
            vel_offset = offset(i) + 6
            W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = IOV_mat

            # Assign Velocity Noise.
            vel_offset = offset(i) + 9
            W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = IV_mat

            # Assign Acceleration Noise.
            acc_offset = offset(i) + 12
            W[acc_offset : acc_offset + 3, acc_offset : acc_offset + 3] = IA_mat
            continue

        offset = lambda i: N * i

        # Assign AngleAxis Noise.
        ori_offset = offset(i)
        W[ori_offset : ori_offset + 3, ori_offset : ori_offset + 3] = O_mat

        # Assign Position Noise.
        pos_offset = offset(i) + 3
        W[pos_offset : pos_offset + 3, pos_offset : pos_offset + 3] = P_mat

        # Assign Angular Velocity Noise.
        vel_offset = offset(i) + 6
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = OV_mat

        # Assign Velocity Noise.
        vel_offset = offset(i) + 9
        W[vel_offset : vel_offset + 3, vel_offset : vel_offset + 3] = V_mat

        # Assign Acceleration Noise.
        acc_offset = offset(i) + 12
        W[acc_offset : acc_offset + 3, acc_offset : acc_offset + 3] = A_mat

    # Required for Covariance calculation.
    Q = W[: N * (K + 1), : N * (K + 1)]
    R = W[N * (K + 1) : (N + M) * (K + 1), N * (K + 1) : (N + M) * (K + 1)]
    return W, Q, R


@lru_cache
def constructSparseStepA(dt: float):
    aa = sp.sparse.coo_array(
        (
            [1, 1, 1, dt, dt, dt],
            (
                [0, 1, 2, 0, 1, 2],
                [0, 1, 2, 6, 7, 8],
            ),
        ),
        shape=(3, 15),
    )

    _a = 0.5 * dt * dt
    p = sp.sparse.coo_array(
        (
            [1, 1, 1, dt, dt, dt, _a, _a, _a],
            (
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [3, 4, 5, 9, 10, 11, 12, 13, 14],
            ),
        ),
        shape=(3, 15),
    )

    av = sp.sparse.coo_array(
        (
            [1, 1, 1],
            (
                [0, 1, 2],
                [6, 7, 8],
            ),
        ),
        shape=(3, 15),
    )

    v = sp.sparse.coo_array(
        (
            [1, 1, 1, dt, dt, dt],
            (
                [0, 1, 2, 0, 1, 2],
                [9, 10, 11, 12, 13, 14],
            ),
        ),
        shape=(3, 15),
    )

    a = sp.sparse.coo_array(
        (
            [1, 1, 1],
            (
                [0, 1, 2],
                [12, 13, 14],
            ),
        ),
        shape=(3, 15),
    )

    # Constant Process matrix
    return sp.sparse.vstack([aa, p, av, v, a], format="coo")


@lru_cache
def constructSparseStepC(M, N):
    return sp.sparse.eye_array(M, n=N, format="coo")


def construct_sparse_model_martix(op, include_cov=False):
    """Construct the Model Matrix H which expresses the process and observation
    model across K+1 measurements(i.e. entire trajectory).
    """

    MK = op.M * (op.K + 1)
    NK = op.N * (op.K + 1)

    # Local A depends on Time delta between samples; for now assumpt sampling is
    # identical between each value.
    A_o = constructSparseStepA(op.dt)
    A_eye = sp.sparse.eye_array(op.N, format="coo")

    # Construct sparse block representation of matrix as list.
    arrs = [[None for _ in range(op.K + 1)] for _ in range(op.K + 1)]
    for i in range(op.K + 1):
        arrs[i][i] = A_eye
    for i in range(op.K):
        arrs[i + 1][i] = -A_o
    A_inv = sp.sparse.block_array(arrs, format="coo")

    # Covariance calculation needs full A matrix. It's expensive to compute so
    # avoid it if not used.
    if include_cov:
        A = sp.sparse.linalg.inv(A_inv.tocsc())
    else:
        A = None
    C = sp.sparse.block_diag(
        [constructStepC(op.M, op.N) for _ in range(op.K + 1)], format="csc"
    )
    # Return H matrix.
    return (sp.sparse.vstack([A_inv, C], format="csc"), A, C)


def construct_sparse_noise_matrix(op, include_cov=False):
    # Noise Matrix. First N*(K+1) is related to motion, next M*(K+1) is related
    # to observation.
    N = op.N
    M = op.M
    K = op.K

    # Initial State
    io_var = np.square(op.iosig)
    ip_var = np.square(op.ipsig)
    iov_var = np.square(op.iovsig)
    iv_var = np.square(op.ivsig)
    ia_var = np.square(op.iasig)

    P_o = np.array(
        3 * [io_var] + 3 * [ip_var] + 3 * [iov_var] + 3 * [iv_var] + 3 * [ia_var],
    )

    # Process
    o_var = np.square(op.osig)
    p_var = np.square(op.psig)
    ov_var = np.square(op.ovsig)
    v_var = np.square(op.vsig)
    a_var = np.square(op.asig)

    Q_k = np.array(
        3 * [o_var] + 3 * [p_var] + 3 * [ov_var] + 3 * [v_var] + 3 * [a_var],
    )

    # Observations
    mo_var = np.square(op.mosig)
    mp_var = np.square(op.mpsig)
    R_k = np.array(3 * [mo_var] + 3 * [mp_var])

    arrs = [None for _ in range(2 * (K + 1))]
    arrs[0] = P_o
    for i in range(1, K + 1):
        arrs[i] = Q_k
    for i in range(K + 1, 2 * (K + 1)):
        arrs[i] = R_k

    W = np.hstack(arrs)

    # Required for Covariance calculation.
    Q = W[: N * (K + 1)] if include_cov else None
    R = W[N * (K + 1) : (N + M) * (K + 1)] if include_cov else None
    return W, Q, R


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Opt:
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

    H, A, C = construct_model_martix(Opt(), include_cov=True)
    sH, sA, sC = construct_sparse_model_martix(Opt(), include_cov=True)
    print(f"C mat diff: {np.linalg.norm(C - sC.toarray())}")
    print(f"A mat diff: {np.linalg.norm(A - sA.toarray())}")
    print(f"H mat diff: {np.linalg.norm(H - sH.toarray())}")

    W, Q, R = construct_noise_matrix(Opt(), include_cov=True)
    sW, sQ, sR = construct_sparse_noise_matrix(Opt(), include_cov=True)

    print(f"R mat diff: {np.linalg.norm(R - np.diag(sR))}")
    print(f"Q mat diff: {np.linalg.norm(Q - np.diag(sQ))}")
    print(f"W mat diff: {np.linalg.norm(W - np.diag(sW))}")
