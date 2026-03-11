import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Write code here
    x = np.array(x,dtype = "float")
    lis = []
    for i in x:
        if i < 0:
            lis.append(i*alpha)
        else:
            lis.append(i)
    lis = np.array(lis,dtype = 'float')
    return lis