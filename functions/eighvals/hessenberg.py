import numpy as np
import scipy.linalg as nla


def hessenberg(A: np.array, calc_q: bool = False) -> tuple[np.array, np.array]:
    """ Compute the Hessenberg decomposition of a square matrix

    Args:
        A (np.array): Matrix to decompose (must be square).
        calc_q (bool, optional): Whether to compute the orthogonal matrix Q. Defaults to False.

    Returns:
        np.array: Hessenberg matrix H.
        np.array: Orthogonal matrix Q. Only returned if calc_q is True.
    """

    H = A.copy()
    m = np.shape(H)[0]
    if calc_q:
        Q = np.eye(m)

    for k in range(1, m):

        # Extraction et copie du vecteur x
        x = H[k:, k-1]
        vk = x.copy()

        # Calcul du vecteur vk
        vk[0] += np.sign(x[0]) * nla.norm(x, 2)
        vk = vk/nla.norm(vk, 2)

        if calc_q:
            # Calcule de la matrice de transformation de Householder hk
            hk = np.eye(m-k) - 2*np.outer(vk, vk)
            # Mise à jour de Q sur les elements concernés
            Q[k:, k:] = np.dot(Q[k:, k:], hk)

        # Multiplication de H à gauche par Qk
        H[k:, k-1:] = H[k:, k-1:] - 2*np.outer(vk, np.dot(vk, H[k:, k-1:]))
        # Multiplication de H à droite par Qk (transposée)
        H[:, k:] = H[:, k:] - 2*np.outer(np.dot(H[:, k:], vk), vk)

    if calc_q:
        return H, Q
    return H