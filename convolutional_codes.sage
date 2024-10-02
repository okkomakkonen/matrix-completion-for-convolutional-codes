from itertools import combinations

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
        X = matrix([x[i:i + n - k] for i in range(0, k * (n - k), n - k)])
    
    if k < n:
        G1 = block_matrix([[X, zero_matrix(k, k)]])
    else:
        G1 = zero_matrix(k, k)
    
    G = block_matrix([[G0, G1], [0, G0]])
    
    return G
                    
def satisfies_condition(G0, X, verbose=False, stop=True):
    """
    Check if G0 and X satisfy the condition for MDP convolutional codes
    
    `satisfies_condition(G0, X)` is equivalent to
    `minor_product_polynomial(G0, X) != 0`, but should be significantly faster
    """
    
    k, n = G0.dimensions()
    G = create_generator_matrix(G0, X)
    
    works = True
    
    for i in range(k + 1):
        for j in range(k, n + 1):
            if i + j != 2*k:
                continue
            
            for I in combinations(range(n), i):
                for J in combinations(range(n, 2*n), j):
                    
                    S = I + J
                    M = G[:, S]
                    if M.rank() != 2*k:
                        
                        works = False
                        
                        if verbose:
                            print("The submatrix on columns", S, "is not invertible")
                        
                        if stop:
                            return works
                    
    return works

def satisfies_condition_fast(G0, X):
    """
    Check if G0 and X satisfy the condition for MDP convolutional codes
    
    `satisfies_condition(G0, X)` is equivalent to
    `minor_product_polynomial(G0, X) != 0`, but should be significantly faster
    """
    
    k, n = G0.dimensions()
    G = create_generator_matrix(G0, X)
    
    if not is_mds_fast(G0):
        return False
    
    works = True
    
    for i in range(k):
        for j in range(k + 1, n + 1):
            if i + j != 2*k:
                continue
            
            for I in combinations(range(n), i):
                for J in combinations(range(n, 2*n), j):
                    
                    S = I + J
                    M = G[:, S]
                    if M.rank() != 2*k:
                        
                        works = False
                        
    return works
        
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

def random_mds_matrix(K, n, m):
    
    while True:
        G = random_matrix(K, n, m)
        if is_mds(G):
            return G
        
def random_systematic_mds_matrix(K, n, m):
    
    while True:
        P = random_matrix(K, n, m - n)
        G = block_matrix([[identity_matrix(n, n), P]])
        if is_mds(G):
            return G

def is_mds(G):
    """Check if the matrix generates an MDS code"""
    
    C = LinearCode(G)
    
    return C.minimum_distance() == C.length() - C.dimension() + 1

def is_mds_fast(G):
    
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
    
    rows = [0..n-1]
    cols = [0..n-1]
    
    rows.remove(row)
    cols.remove(col)
    
    if verbose:
        print(A[rows, cols])
    
    return A[rows, cols].det()


def cofactor_expansion(A, row=None, col=None):
    """Compute the cofactors for a given row or column of the matrix A"""
    
    if (row is None and col is None) or (row is not None and col is not None):
        raise
        
    assert A.is_square()
    n, _ = A.dimensions()
        
    if row is None:
        
        j = col
        
        return [(-1)^(i + j) * minor(A, i, j) * A[i, j] for i in range(n)]
    
    if col is None:
        
        i = row
        
        return [(-1)^(i + j) * minor(A, i, j) * A[i, j] for j in range(n)]
