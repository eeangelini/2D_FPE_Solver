import numpy as np
from make_tridiag import make_tridiag

class structtype():
    '''Dummy class to create a structtype object'''
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val

def make_matrix(M1 = None,N1 = None,M2 = None,N2 = None,x1_dt = None,x2_dt = None,D = None): 
    '''[M0,Md,Am,x]=make_matrix(M,N); creates matrices to solve advection-diffusion problem
        INPUTS: 
            - M specifies the spatial domain x1=(-M1,M1), x2=(-M2,M2)
            - N1,N2 are the grid size (size of matrices (N1*N2)x(N*N2) and vectors (N1*N2)x1)
        OUTPUT: 
            - D1,D2,D11,D22,x1,x2 are used to solve eqn
    '''
    
    #params
    dx1 = 2 * M1 / (N1 - 1)
    if (dx1 < 0 or dx1 > 0.5):
        raise ValueError('Make M1 smaller and/or N1 larger!!')
    
    x1 = np.linspace(-M1,M1,N1)
    if dx1 != x1[1] - x1[0]:
        raise ValueError('Initializing x1 failed!!')
    
    dx2 = 2 * M2 / (N2 - 1)
    if (dx2 < 0 or dx2 > 0.5):
        raise ValueError('Make M2 smaller and/or N2 larger!!')
    
    x2 = np.linspace(-M2,M2,N2)
    if dx2 != x2[1] - x2[0]:
        raise ValueError('Initializing x2 failed!!')
    
    #---- Initialize D1,D2,D11,D22 "structtypes" (analog to Matlab struct) ----
    D1 = structtype()
    D2 = structtype()
    D11 = structtype()
    D22 = structtype()
    
    a = []
    b = np.transpose(np.zeros((N1 * N2,1)))
    c = []
    for i in range(N2):
        a_coeff = 0
        c_coeff = 0
        for k in range(x1_dt.shape[0]):
            a_coeff = a_coeff - np.multiply((- x1_dt[k,1]) * (x2[i] ** x1_dt[k,3]),(x1[:-1] ** x1_dt[k,2])) / (2 * dx1)
            c_coeff = c_coeff + np.multiply((- x1_dt[k,1]) * (x2[i] ** x1_dt[k,3]),(x1[1:] ** x1_dt[k,2])) / (2 * dx1)
        a = np.array([a,0,np.transpose(a_coeff)])
        c = np.array([c,np.transpose(c_coeff),0])
    
    D1.a = a
    D1.b = b
    D1.c = c
    D1.matrix = make_tridiag(a,b,c)
    ###########################
    
    a = []
    b = np.transpose(np.zeros((N1 * N2,1)))
    c = []
    for i in range(N1):
        a_coeff = 0
        c_coeff = 0
        for k in range(x2_dt.shape[0]):
            a_coeff = a_coeff - np.multiply((- x2_dt[k,1]) * (x2[:-1] ** x2_dt[k,3]),(x1(i) ** x2_dt[k,2])) / (2 * dx2)
            c_coeff = c_coeff + np.multiply((- x2_dt[k,1]) * (x2[1:] ** x2_dt[k,3]),(x1(i) ** x2_dt[k,2])) / (2 * dx2)
        a = np.array([a,0,np.transpose(a_coeff)])
        c = np.array([c,np.transpose(c_coeff),0])
    
    D2.a = a
    D2.b = b
    D2.c = c
    D2.matrix = make_tridiag(a,b,c)
    ###########################
    
    a = []
    b = []
    c = []
    for i in np.arange(1,N2+1).reshape(-1):
        a = np.array([a,0,D(1) * np.transpose((np.ones((N1 - 1,1)) / (dx1 * dx1)))])
        c = np.array([c,D(1) * np.transpose((np.ones((N1 - 1,1)) / (dx1 * dx1))),0])
        b = np.array([b,- 2 * D(1) * np.transpose((np.ones((N1,1)) / (dx1 * dx1)))])
    
    D11.a = a
    D11.b = b
    D11.c = c
    D11.matrix = make_tridiag(a,b,c)
    ###########################
    
    a = []
    b = []
    c = []
    for i in np.arange(1,N1+1).reshape(-1):
        a = np.array([a,0,D(2) * np.transpose((np.ones((N2 - 1,1)) / (dx2 * dx2)))])
        c = np.array([c,D(2) * np.transpose((np.ones((N2 - 1,1)) / (dx2 * dx2))),0])
        b = np.array([b,- 2 * D(2) * np.transpose((np.ones((N2,1)) / (dx2 * dx2)))])
    
    D22.a = a
    D22.b = b
    D22.c = c
    D22.matrix = make_tridiag(a,b,c)
    return D1,D2,D11,D22,x1,x2