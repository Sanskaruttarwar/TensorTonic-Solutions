import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    n = len(A)
    m = len(A[0])
    ret = np.empty((m,n))
    
    for i in range(n):
        for j in range(m):
            ret[j][i] = A[i][j]

    return ret
    
