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
    
    dx2 = 2 * M2 / (N2 - 1)
    if (dx2 < 0 or dx2 > 0.5):
        raise ValueError('Make M2 smaller and/or N2 larger!!')
    
    x2 = np.linspace(-M2,M2,N2)
    
    #---- Initialize D1,D2,D11,D22 "structtypes" (analog to Matlab struct) ----
    D1 = structtype()
    D2 = structtype()
    D11 = structtype()
    D22 = structtype()
    
    for i in range(N2):
        a_coeff = 0
        c_coeff = 0
        for k in range(x1_dt.shape[0]):
            a_coeff = a_coeff - np.multiply((- x1_dt[k,0]) * (x2[i] ** x1_dt[k,2]),(x1[:-1] ** x1_dt[k,1])) / (2 * dx1)
            c_coeff = c_coeff + np.multiply((- x1_dt[k,0]) * (x2[i] ** x1_dt[k,2]),(x1[1:] ** x1_dt[k,1])) / (2 * dx1)
        if i == 0:
            a = np.concatenate((np.zeros(1),a_coeff),axis = 0)
            c = np.concatenate((c_coeff,np.zeros(1)),axis = 0)
        else:
            a = np.concatenate((a,np.zeros(1),a_coeff),axis = 0)
            c = np.concatenate((c,c_coeff,np.zeros(1)),axis = 0)
    
    D1.a = a
    D1.b = np.zeros(N1 * N2)
    D1.c = c
    D1.matrix = make_tridiag(D1.a,D1.b,D1.c)
    ###########################
    
    for i in range(N1):
        a_coeff = 0
        c_coeff = 0
        for k in range(x2_dt.shape[0]):
            a_coeff = a_coeff - np.multiply((- x2_dt[k,0]) * (x2[:-1] ** x2_dt[k,2]),(x1[i] ** x2_dt[k,1])) / (2 * dx2)
            c_coeff = c_coeff + np.multiply((- x2_dt[k,0]) * (x2[1:] ** x2_dt[k,2]),(x1[i] ** x2_dt[k,1])) / (2 * dx2)
        if i==0:
            a = np.concatenate((np.zeros(1),a_coeff),axis = 0)
            c = np.concatenate((c_coeff,np.zeros(1)),axis = 0)
        else:
            a = np.concatenate((a,np.zeros(1),a_coeff),axis = 0)
            c = np.concatenate((c,c_coeff,np.zeros(1)),axis = 0)
    
    D2.a = a
    D2.b = np.zeros(N1 * N2)
    D2.c = c
    D2.matrix = make_tridiag(D2.a,D2.b,D2.c)
    ###########################
    
    for i in range(N2):
        if i == 0:
            a = np.concatenate((np.zeros(1),D[0]*np.ones(N1-1) / (dx1 * dx1)),axis=0)
            c = np.concatenate((D[0]*np.ones(N1-1) / (dx1 * dx1),np.zeros(1)),axis=0)
            b = -2 * D[0] * np.ones(N1) / (dx1 * dx1)
        else:
            a = np.concatenate((a,np.zeros(1),D[0]*np.ones(N1-1) / (dx1 * dx1)),axis=0)
            c = np.concatenate((c,D[0]*np.ones(N1-1) / (dx1 * dx1),np.zeros(1)),axis=0)
            b = np.concatenate((b,-2 * D[0] * np.ones(N1) / (dx1 * dx1)),axis=0)
    
    D11.a = a
    D11.b = b
    D11.c = c
    D11.matrix = make_tridiag(D11.a,D11.b,D11.c)
    ###########################
    
    for i in range(N1):
        if i == 0:
            a = np.concatenate((np.zeros(1),D[1]*np.ones(N2-1) / (dx2 * dx2)),axis=0)
            c = np.concatenate((D[1]*np.ones(N2-1) / (dx2 * dx2),np.zeros(1)),axis=0)
            b = -2 * D[1] * np.ones(N2) / (dx2 * dx2)
        else:
            a = np.concatenate((a,np.zeros(1),D[1]*np.ones(N2-1) / (dx2 * dx2)),axis=0)
            c = np.concatenate((c,D[1]*np.ones(N2-1) / (dx2 * dx2),np.zeros(1)),axis=0)
            b = np.concatenate((b,-2 * D[1] * np.ones(N2) / (dx2 * dx2)),axis=0)
    
    D22.a = a
    D22.b = b
    D22.c = c
    D22.matrix = make_tridiag(D22.a,D22.b,D22.c)
    return D1,D2,D11,D22,x1,x2