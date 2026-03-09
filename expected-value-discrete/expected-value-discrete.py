import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.array(x, dtype = float)
    p = np.array(p, dtype = float)

    if x.shape != p.shape:
        raise ValueError("Shape of x and p not matched")

    if not np.allclose(np.sum(p), 1):
        raise ValueError("Probabilites must sum to 1")

    return float(np.sum(x*p))

    
