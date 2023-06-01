import numpy as np

from functions.eighvals.gershgorin import gershgorin_trig
from functions.eighvals.bissection import bissection
from functions.eighvals.puissance_inverse import puissance_inverse
from functions.svd.bidiagonal import bidiagonal, reflecteur


def svd(A: np.array) -> tuple[np.array, np.array, np.array]:
    """ Compute the singular values and singular vectors of a matrix A.

    Args:
        A (np.array): Matrix to decompose.

    Returns:
        tuple[np.array, np.array, np.array]: U, Sigma, Vt such that A = U * Sigma * Vt.
    """    

    m, n = np.shape(A)

    B, VL, VR = bidiagonal(A) # Calcul de la décomposition bidiagonale de A

    # Calcul de la matrice de bloc H
    B = B[0:n, 0:n]
    H = np.zeros([2*n, 2*n], dtype=np.float64)
    H[0:n, n:2*n] = np.transpose(B)
    H[n:2*n, 0:n] = B

    # On créé une autre matrice de bloc P à l'aide d'éléments d'identité
    # On fabrique ensuite la matrice T, qui est la version transformée de H.
    P = np.zeros([2*n, 2*n], dtype=np.float64)
    for i in range(0, n):
        P[i, 2*i] = 1
        P[n+i, 2*i+1] = 1
    T = np.dot(np.transpose(P), np.dot(H, P))

    # On calcule les valeurs propres de T par la méthode de Gershgorin et de la bissection
    bmin, bsup = gershgorin_trig(T)
    bmin -= 0.0001
    bsup += 0.0001
    eigvals = bissection(T, bmin, bsup)
    Lambda = eigvals[n:2*n]

    # On calcule les vecteurs propres de T par la méthode de la puissance inverse
    eigvecs = np.zeros([2*n, 2*n], dtype=np.float64)
    for (i, l) in enumerate(eigvals):
        eigvecs[:, i] = puissance_inverse(T, l)


    Q = eigvecs[:, n:2*n]
    Y = np.sqrt(2) * np.dot(P, Q)
    newVt = np.transpose(Y[0:n, :])

    newU = np.zeros([m, n], dtype=np.float64)
    newU[0:n, :] = Y[n:2*n, :]

    # On extrait les valeurs singulières
    newSigma = np.array(np.diag(Lambda), dtype=np.float64)

    # On calcule les vecteurs singuliers gauches
    for i in range(n-1, -1, -1):
        Q = reflecteur(m, VL[i])
        newU = np.dot(Q, newU)

    # On calcule les vecteurs singuliers droits
    for i in range(n-3, -1, -1):
        Q = reflecteur(n, VR[i])
        newVt = np.dot(newVt, Q)
    
    return newU, newSigma, newVt
