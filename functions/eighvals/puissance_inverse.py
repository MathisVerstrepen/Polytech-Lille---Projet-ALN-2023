import numpy as np
import scipy.linalg as nla


def solve_triangular(A: np.array, b: np.array, lower: bool, overwrite_b: bool, unit_diagonal: bool = False):
    """ Solve a linear system of equations with a triangular matrix.

    Args:
        A (np.array): Triangular matrix. A from Ax = b.
        b (np.array): Vector of constants. b from Ax = b.
        lower (bool): If True, the matrix is lower triangular, otherwise it is upper triangular.
        overwrite_b (bool): If True, the function overwrites the vector b.
        unit_diagonal (bool): If True, the diagonal of the matrix is assumed to be 1. Defaults to False.

    Returns:
        np.array: Solution of the linear system of equations.
    """

    if overwrite_b:
        x = b
    else:
        x = b.copy()

    if lower:
        for i in range(0, A.shape[0]):
            for j in range(0, i):
                x[i] -= A[i, j] * x[j]
            if not unit_diagonal:
                x[i] /= A[i, i]
    else:
        for i in range(A.shape[0]-1, -1, -1):
            for j in range(i+1, A.shape[1]):
                x[i] -= A[i, j] * x[j]
            if not unit_diagonal:
                x[i] /= A[i, i]

    return x


def lu_factor(A: np.array) -> tuple[np.array, np.array]:
    """ Compute the LU factorization of a matrix A.

    Args:
        A (np.array): Matrix to factorize.

    Returns:
        tuple[np.array, np.array]: Tuple containing the factorized matrix and the pivot vector.
    """
    
    N = A.shape[0]
    U = (A.copy()).astype('float64')
    L = (np.zeros((N,N))).astype('float64')
    piv = np.array([], dtype=int)

    for n in range(0, N):
        #Recherche du pivot
        pivot_value = 0
        pivot_index = 0
        
        for i in range(n, N):
            if abs(U[i, n]) > pivot_value:
                pivot_value = abs(U[i, n])
                pivot_index = i
                
        piv = np.append(piv, pivot_index)
        
        #Si le pivot n'est pas celui par défaut, on intervertit les lignes
        if pivot_index != n:
            U[[n,pivot_index]] = U[[pivot_index,n]]
            L[[n, pivot_index], :n] = L[[pivot_index,n], :n]
            
        #Calculs classiques de la facto LU
        for i in range(n+1, N):
            L[i,n] = U[i,n] / U[n,n]
            for j in range(n,N):
                U[i,j] -= L[i,n] * U[n,j]

    return (L+U, piv)


def puissance_inverse(A: np.array, mu: np.float64, epsilon: np.float32 = np.finfo(np.float32).eps):
    """ Compute the eigenvector of a matrix A using the inverse method and the given eigenvalue mu.

    Args:
        A (np.array): Matrix to compute the eigenvector of.
        mu (np.float64): One of the eigenvalues of A.
        epsilon (np.float32): Precision of the computation.

    Returns:
        np.array: Eigenvector of A corresponding to the eigenvalue mu.
    """

    n = A.shape[0]
    G_A = A - mu*np.eye(n)
    G_A, piv = lu_factor(G_A)

    v = np.random.rand(n)  # On prend un vecteur aléatoire
    while (abs(np.dot(v, np.dot(A, v)) - mu) > epsilon):
        for i in range(0, n):
            if i != piv[i]:
                v[i], v[piv[i]] = v[piv[i]], v[i]

        solve_triangular(G_A, v, lower=True,
                         unit_diagonal=True, overwrite_b=True)
        solve_triangular(G_A, v, lower=False, overwrite_b=True)

        v = (1/nla.norm(v, 2)) * v

    return v