import numpy as np


def sturm_seq(A: np.array, mu: int) -> np.array:
    """ Compute the Sturm sequence of a matrix A.

    Args:
        A (np.array): Matrix to compute the Sturm sequence of.
        mu (int): Adjusts the matrix A by subtracting mu from the diagonal.

    Returns:
        np.array: Sturm sequence of A.
    """

    m = A.shape[0]
    T = np.empty(m+1)  # Tableau des déterminants
    T[0] = 1

    T[1] = A[0, 0] - mu
    T[2] = (A[1, 1] - mu) * T[1] - A[0, 1]**2
    for k in range(2, m):
        T[k+1] = (A[k, k] - mu) * T[k] - (A[k, k-1]**2) * T[k - 1]

    return T


def count_sign_changes(seq: np.array) -> int:
    """ Compute the number of sign changes in a Sturm sequence.

    Args:
        seq (np.array): Sequence of determinants.

    Returns:
        int: Number of sign changes.
    """

    count = 0
    sign_seq = np.sign(seq)  # Tableau des signes
    for i in range(0, len(seq)-1):
        if sign_seq[i+1] == 0:
            sign_seq[i+1] = sign_seq[i]
        elif sign_seq[i] != sign_seq[i+1]:
            count += 1

    return count


def bissection(A: np.array, binf: np.float64, bsup: np.float64, precision: np.float32 = np.finfo(np.float32).eps) -> np.array:
    """ Compute the eigenvalues of a matrix A using the bissection method. 
        Recursively calls itself until the interval is small enough.

    Args:
        A (np.array): Matrix to compute the eigenvalues of.
        binf (np.float64): Inferior bound of the interval.
        bsup (np.float64): Superior bound of the interval.
        e (np.float32): Precision of the computation. Defaults to np.finfo(np.float32).eps (machine precision in 32 bits).

    Returns:
        np.array: Eigenvalues of A in the interval [binf, bsup]. 
                  Empty array if the interval does not contain any eigenvalue.
    """

    n = count_sign_changes(sturm_seq(A, bsup)) - \
        count_sign_changes(sturm_seq(A, binf))
    if n == 0:
        return np.array([])
    if bsup - binf > precision:
        m = (binf+bsup)/2
        return np.concatenate((bissection(A, binf, m, precision), bissection(A, m, bsup, precision)), axis=None)
    else:
        return np.array([(binf + bsup)/2])
