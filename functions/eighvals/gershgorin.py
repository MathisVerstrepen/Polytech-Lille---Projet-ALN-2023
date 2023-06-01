import numpy as np


def gershgorin_trig(C: np.array) -> tuple[np.float64, np.float64]:
    """ Compute an interval containing all eigenvalues of a tridiagonal matrix

    Args:
        C (np.array): Tridiagonal matrix.

    Returns:
        np.float64: Lower bound of the interval, obtained via Gershgorin circle theorem
        np.float64: Upper bound of the interval, obtained via Gershgorin circle theorem
    """
    dim = np.shape(C)[0]-1

    # Initialisation de Inf et Sup aux plus petites et plus grandes valeurs possibles
    Inf = min(C[0, 0]-abs(C[0, 1]), C[dim, dim]-abs(C[dim, dim-1]))
    Sup = max(C[0, 0]+abs(C[0, 1]), C[dim, dim]+abs(C[dim, dim-1]))

    # Calcul de Inf et Sup via le théorème des cercles de Gershgorin
    for i in range(1, dim):
        a = C[i, i] - abs(C[i, i-1]) - abs(C[i, i+1])
        b = C[i, i] + abs(C[i, i-1]) + abs(C[i, i+1])
        if a < Inf:
            Inf = a
        if b > Sup:
            Sup = b

    return Inf, Sup
