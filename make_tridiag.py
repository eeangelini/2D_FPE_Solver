import numpy as np
from scipy.sparse import csr_matrix as sparse

def make_tridiag(a = None,b = None,c = None): 
    # Output a sparse tridiagonal matrix given a, b, c
    
    N = len(a)
    if (len(b) != N or len(c) != N):
        raise ValueError('Make sure the length of a, b, c are the same...\n' % ())
    
    x_indx = np.concatenate([np.arange(1,N),np.arange(N),np.arange(N-1)],axis=0)
    y_indx = np.concatenate([np.arange(N-1),np.arange(N),np.arange(1,N)],axis=0)
    eles = np.concatenate([a[1:],b,c[:-1]],axis=0)
    Mat = sparse((eles,(x_indx,y_indx)),shape = (N,N))
    return Mat
