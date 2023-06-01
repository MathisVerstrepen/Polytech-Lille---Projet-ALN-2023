import numpy as np

from functions.eighvals.gershgorin import gershgorin_trig
from functions.eighvals.bissection import bissection
from functions.eighvals.hessenberg import hessenberg
from functions.eighvals.puissance_inverse import puissance_inverse

def eigh( A : np.array ) -> tuple[np.array, np.array]:
    """ Compute the eigenvalues and eigenvectors of a symmetric matrix A.
        Same as numpy.linalg.eigh

    Args:
        A (np.array): Symmetric matrix to compute the eigenvalues and eigenvectors.

    Returns:
        tuple[np.array, np.array]: Eigenvalues and eigenvectors of A.
    """
    
    n = np.shape(A)[0]                  # Dimension of the matrix
    H = hessenberg(A)                   # Hessenberg form of A
    b_inf, b_sup = gershgorin_trig(H)   # Bounds of the eigenvalues of H
    w = bissection(H, b_inf, b_sup)     # Eigenvalues of H

    # Compute the eigenvectors of A
    v = np.empty([n, n], dtype=np.float64)
    for (idx, eig) in enumerate(w):
        v[:, idx] = puissance_inverse(A, eig)
        
    return w, v