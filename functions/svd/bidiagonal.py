import numpy as np
import scipy.linalg as nla


def reflecteur(p: int, v: np.array) -> np.array:
    """ Compute the reflection matrix associated to a vector v.

    Args:
        p (int): Number of rows of the matrix.
        v (np.array): Vector to compute the reflection matrix.

    Returns:
        np.array: Reflection matrix associated to v.
    """

    n = v.shape[0]
    F = np.eye(n) - 2 * np.outer(v, v)
    Q = np.eye(p, dtype=np.float64)
    Q[p-n:p, p-n:p] = F

    return Q


def bidiagonal(A: np.array) -> tuple[np.array, list[np.array], list[np.array]]:
    """ Compute the bidiagonal decomposition of a matrix A.

    Args:
        A (np.array): Matrix to decompose.

    Returns:
        tuple[np.array, list[np.array], list[np.array]]: B, VL, VR such that A = VL * B * VR.
    """

    B = np.copy(A)
    m, n = np.shape(B)
    VL = []
    VR = []

    for k in range(0, n):

        x = B[k:m, k]
        vk = np.copy(x)

        vk[0] = vk[0] + np.sign(vk[0]) * nla.norm(x, 2)
        vk = vk/nla.norm(vk, 2)
        VL.append(vk)

        Q = reflecteur(m, vk)
        B = np.dot(Q, B)

        if k < n - 2:
            x = B[k, k+1:n+1]
            vk = np.copy(x)
            vk[0] = vk[0] + np.sign(vk[0]) * nla.norm(x, 2)
            vk = vk/nla.norm(vk, 2)
            VR.append(vk)

            Q = reflecteur(n, vk)
            B = np.dot(B, Q)

    return B, VL, VR
