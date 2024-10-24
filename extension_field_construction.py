from itertools import permutations

from sage.all import *

from convolutional_code_utils import *


def create_X_matrix(beta, k):
    """Create the X matrix for this construction

    If beta = [b1, b2, b3] and k = 4, then this method returns the matrix

        0  0  0
        b1 0  0
        b2 b1 0
        b3 b2 b1
    """

    c = len(beta)

    assert k >= c, "dimension k has to be at least as large as the codimension n - k"

    Z = zero_matrix(k - c, c)
    T = matrix.toeplitz(beta, zero_vector(c - 1))

    return Z.stack(T)


def test_all_betas(n, k, q=None):
    """
    This function will test all constructions of the given form where the elements
    beta are powers of z (the generator of the extension field)
    """

    c = n - k

    assert k >= c

    if not q:
        q = next_prime(n - 1)

    assert is_prime(q), "q must be prime"

    K = GF(q)
    K_ext = K.extension(c, "z")
    (z,) = K_ext.gens()

    G0 = vandermonde_matrix(K, k, n)

    elements = [z**i for i in range(c)]

    for beta in permutations(elements):
        X = create_X_matrix(beta, k)
        G0X = block_matrix([[G0, X]])

        if is_mds(G0X) and satisfies_condition_fast(G0, X):
            print(beta, "works")
        else:
            print(beta, "does not work")
