import numpy as np
from scipy.sparse import csr_matrix as sparse, eye as speye
from scipy.stats import multivariate_normal as mvn

def TDMA(a, b, c, d):
    '''Solving tridiagonal linear system of equations Tridiag(a,b,c) @ y = d (Thomas Algorithm).
    '''
    n = len(d)
    w = np.zeros(n-1,float)
    g = np.zeros(n, float)
    y = np.zeros(n,float)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    # first row of coefficients
    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    y[n-1] = g[n-1]

    # backsubstitution
    for i in range(n-1,0,-1):
        y[i-1] = g[i-1] - w[i-1]*y[i]
    return y
    
def solve_advDif(D1 = None,D2 = None,D11 = None,D22 = None,x1 = None,x2 = None,dt = None,T_end = None,mu = None,sigma = None): 
    t = np.transpose((np.arange(0,T_end+dt,dt)))
    N1 = len(x1)
    N2 = len(x2)
    p = np.zeros((N1,N2,len(t)))
    #mu = [0,10]; sigma = 1/2*eye(2);
    X1,X2 = np.meshgrid(x1,x2)
    # initialize with multivariate normal distribution
    p0 = np.reshape(mvn(mu,sigma).pdf(np.dstack((X1,X2))),(N2,N1)).T
    p[:,:,0] = p0.copy()
    #***done setting initial condition***
    
    # reshape for iterating
    p = np.reshape(p,(N1 * N2,len(t)))
    #----create matrix 'A' and vector 'b' here (Diffusion+Drift)
    rhs_first = sparse(speye(N1 * N2) + 0.5 * dt * (D1.matrix + D11.matrix))
    rhs_second = sparse(speye(N1 * N2) + 0.5 * dt * (D2.matrix + D22.matrix))
    a_first = - 0.5 * dt * (D2.a + D22.a)
    b_first = np.ones(N1 * N2) - 0.5 * dt * (D2.b + D22.b)
    c_first = - 0.5 * dt * (D2.c + D22.c)
    a_second = - 0.5 * dt * (D1.a + D11.a)
    b_second = np.ones(N1 * N2) - 0.5 * dt * (D1.b + D11.b)
    c_second = - 0.5 * dt * (D1.c + D11.c)
    for j in range(1,len(t)):
        if j%100 == 0:
            print('Number of Iteration %d out of %d finished...\n' % (j,len(t)))
        #----solve for p[:,j]=....
        reshaped_r = np.reshape(rhs_first * p[:,j - 1],(N1,N2)).T.flatten()
        p_half = TDMA(a_first,b_first,c_first,reshaped_r)
        reshaped_r = np.reshape(rhs_second * p_half,(N2,N1)).T.flatten()
        p[:,j] = TDMA(a_second,b_second,c_second,reshaped_r)
    
    p = np.reshape(p,(N1,N2,len(t)))
    return t,p