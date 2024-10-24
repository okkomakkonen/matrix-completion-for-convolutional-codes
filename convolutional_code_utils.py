"""
This file contains some helper methods that are used in all of the notebooks.

The generator matrices for the convolutional codes are specified by the pair (G0, X)
such that the 1st truncated sliding generator matrix is

     G0 | X 0
    ---------
      0 | G0

"""

from itertools import combinations

from sage.all import *


def create_generator_matrix(G0, X=None):
    """
    Create the generator matrix using G0 and X

    If X is not specified, it is set to contain indeterminates.
    """

    k, n = G0.dimensions()
    K = G0.base_ring()

    # if X is not provided, then it is set to contain k(n - k) indeterminates
    # the indexing here is zero based
    if X is None and k < n:
        R = PolynomialRing(K, k, (n - k), var_array="x")
        x = R.gens()
        X = matrix([x[i : i + n - k] for i in range(0, k * (n - k), n - k)])

    if k < n:
        G1 = block_matrix([[X, zero_matrix(k, k)]])
    else:
        G1 = zero_matrix(k, k)

    O = zero_matrix(G0.base_ring(), k, n)

    G = block_matrix([[G0, G1], [O, G0]])

    return G


def satisfies_condition(G0, X, verbose=False, stop_early=True):
    """
    Check if G0 and X satisfy the condition for MDP convolutional codes

    If `verbose` is set to True, then this method will print out any minors that are zero.
    If `stop_early` is set to False, then this method will continue after the first zero minor.
    """

    k, n = G0.dimensions()
    G = create_generator_matrix(G0, X)

    satisfies = True

    for i in range(k + 1):
        for j in range(k, n + 1):
            if i + j != 2 * k:
                continue

            for I in combinations(range(n), i):
                for J in combinations(range(n, 2 * n), j):
                    S = I + J
                    M = G[:, S]
                    if M.rank() != 2 * k:
                        satisfies = False

                        if verbose:
                            print("The submatrix on columns", S, "is not invertible")

                        if stop_early:
                            return satisfies

    return satisfies


def satisfies_condition_fast(G0, X):
    """
    Check if G0 and X satisfy the condition for MDP convolutional codes

    Assumes that G0X generates an MDS code.
    """

    k, n = G0.dimensions()
    G = create_generator_matrix(G0, X)

    for i in range(k):
        for j in range(n - k + 1):
            for l in range(k):
                if i + j + l != 2 * k:
                    continue

                for I in combinations(range(n), i):
                    for J in combinations(range(n, 2 * n - k), j):
                        for L in combinations(range(2 * n - k, 2 * n), l):
                            S = I + J + L
                            M = G[:, S]
                            if M.rank() != 2 * k:
                                return False

    return True


def cauchy_matrix(K, n, m):
    """
    Create a n x m Cauchy matrix over the field K

    The field K has to have at least n + m elements.
    """

    assert K.order() >= n + m

    return matrix(n, m, lambda i, j: 1 / (K.list()[i] - K.list()[-j - 1]))


def vandermonde_matrix(K, n, m):
    """
    Create a n x m Vandermonde matrix over the field K

    The field K has to have at least m elements.
    """

    assert K.order() >= m

    return matrix.vandermonde(K.list()[:m]).T[:n]


def extended_vandermonde(K, n, m):
    """
    Create a n x m extended Vandermonde matrix over the field K

    The first m - 1 columns are a regular Vandermonde matrix while
    the last column is (0, ..., 0, 1). The field has to have at
    least m - 1 elements.
    """

    M = vandermonde_matrix(K, n, m - 1)

    c = vector([0] * (n - 1) + [1])

    return matrix(M.columns() + [c]).T


def random_mds_matrix(K, n, m):
    """Finds a random MDS matrix by brute force search"""

    while True:
        G = random_matrix(K, n, m)
        if is_mds(G):
            return G


def random_systematic_mds_matrix(K, n, m):
    """Finds a random MDS matrix in systematic form by brute force search"""

    I = identity_matrix(n, n)

    while True:
        P = random_matrix(K, n, m - n)
        G = block_matrix([[I, P]])
        if is_mds(G):
            return G


def is_mds(G):
    """Check if the matrix generates an MDS code"""

    # NOTE: minimum distance calculation is not implemented for large fields, so we have to check manually
    try:
        C = LinearCode(G)

        return C.minimum_distance() == C.length() - C.dimension() + 1

    except NotImplementedError:
        k, n = G.dimensions()

        for I in combinations(range(n), k):
            M = G[:, I]
            if M.rank() != k:
                return False

        return True


def minor(A, row, col, verbose=False):
    """Compute the minor for given row and column of the matrix A"""

    assert A.is_square()

    n, _ = A.dimensions()

    rows = list(range(n))
    cols = list(range(n))

    rows.remove(row)
    cols.remove(col)

    if verbose:
        print(A[rows, cols])

    return A[rows, cols].det()


def cofactor_expansion(A, row=None, col=None):
    """Compute the cofactors for a given row or column of the square matrix A"""

    if (row is None and col is None) or (row is not None and col is not None):
        raise ValueError("must specify exactly one of row or col")

    if not A.is_square():
        raise ValueError("matrix is not square")

    n, _ = A.dimensions()

    if row is None:
        j = col

        return [Integer(-1) ** (i + j) * minor(A, i, j) * A[i, j] for i in range(n)]

    if col is None:
        i = row

        return [Integer(-1) ** (i + j) * minor(A, i, j) * A[i, j] for j in range(n)]
