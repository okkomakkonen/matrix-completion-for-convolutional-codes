from itertools import combinations

from sage.all import *

from convolutional_code_utils import *


def minor_product_polynomial(G0, X=None):
    """
    Compute the value of the minor product polynomial for G0 at X

    If X is not specified, it is set to contain indeterminates.
    """

    k, n = G0.dimensions()
    G = create_generator_matrix(G0, X)
    P = 1

    for i in range(k + 1):
        for j in range(k, n + 1):
            if i + j != 2 * k:
                continue

            for I in combinations(range(n), i):
                for J in combinations(range(n, 2 * n), j):
                    # combine I and J as a union
                    S = I + J
                    M = G[:, S]
                    P *= M.det()

    return P


def minor_product_polynomial_degree(n, k):
    """Compute the degree of the minor product polynomial for any k x n MDS matrix G0"""

    assert 0 <= k <= n

    d = 0
    for i in range(k + 1):
        for j in range(k, n + 1):
            if i + j != 2 * k:
                continue

            d += binomial(n, i) * binomial(n, j) * (j - k)

    return d


def minor_product_polynomial_individual_degree(n, k):
    """
    Compute (an upper bound on) the individual degree of any variable
    in the minor product polynomial of any k x n MDS matrix G0
    """

    assert 0 <= k <= n

    d = 0
    for i in range(k):
        for j in range(k + 1, n + 1):
            if i + j != 2 * k:
                continue

            d += binomial(n, i) * binomial(n - 1, j - 1)

    return d


def minor_product_polynomial_individual_degree_standard_form(n, k):
    """
    Compute (an upper bound on) the individual degree of any variable
    in the minor product polynomial of any k x n MDS matrix G0 that is
    in standard form
    """

    assert 0 <= k <= n

    d = 0
    for i in range(k):
        for j in range(k + 1, n + 1):
            if i + j != 2 * k:
                continue

            d += binomial(n - 1, i) * binomial(n - 1, j - 1)

    return d


def print_minors(G0):
    """Compute the degree of the minor product polynomial for MDS matrix G0"""

    k, n = G0.dimensions()
    G = create_generator_matrix(G0)

    print(G)
    print()

    for i in range(k + 1):
        for j in range(k, n + 1):
            if i + j != 2 * k:
                continue

            for I in combinations(range(n), i):
                for J in combinations(range(n, 2 * n), j):
                    S = I + J
                    M = G[:, S]

                    # do sanity checks
                    assert M.det().degree() == j - k
                    assert M.det().is_homogeneous()

                    print(f"{I} + {J}: {M.det()}")


def find_random_solution(G0, num_tries=10_000, seed=1):
    """
    Find a solution for X by using random sampling

    This will only do 10,000 tries before raising an exception.
    The random seed is set so that it produces reproducible results.
    """

    set_random_seed(seed)

    k, n = G0.dimensions()
    K = G0.base_ring()

    for _ in range(num_tries):
        X = random_matrix(K, k, (n - k))

        if satisfies_condition(G0, X):
            G = create_generator_matrix(G0, X)
            return G

    raise Exception(f"did not find a solution in {num_tries} tries")


def find_solution(G0, K=None):
    """Find a solution for X by going through all possibilities"""

    k, n = G0.dimensions()
    if K is None:
        K = G0.base_ring()

    for X in MatrixSpace(K, k, (n - k)):
        if satisfies_condition(G0, X):
            G = create_generator_matrix(G0, X)
            return G

    raise Exception(f"solution does not exist over this field")


def print_all_solutions(G0, K=None):
    """Print all solutions for X by going through all possibilities"""

    k, n = G0.dimensions()
    if K is None:
        K = G0.base_ring()

    found = False
    for X in MatrixSpace(K, k, (n - k)):
        if satisfies_condition_fast(G0, X):
            G = create_generator_matrix(G0, X)
            found = True
            print(G)
            print()

    if not found:
        print("finished without finding any solutions")
