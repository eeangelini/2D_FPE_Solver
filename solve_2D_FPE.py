import numpy as np
from make_matrix import make_matrix
from solve_advDif import solve_advDif

# #{
# # heat diffusion
# x1_dt = np.array([0,0,0])
# x2_dt = np.array([0,0,0])
# D = np.array([0.5,0.5])
# dt = 0.01
# T_end = 25
# M1 = 10
# M2 = 10
# N1 = 200
# N2 = 200
# mu = np.array([0,0])
# sigma = 1 / 5 * np.eye(2)
# #}
# #{
# # linear oscillator
# x1_dt = np.array([1,0,1])
# x2_dt = np.array([[- 0.2,0,1],[- 1,1,0]])
# D = np.array([0,0.2])
# dt = 0.01
# T_end = 25
# M1 = 10
# M2 = 10
# N1 = 200
# N2 = 200
# mu = np.array([5,5])
# sigma = 1 / 9 * np.eye(2)
# #}
# #{
# # bimodal oscillator
# x1_dt = np.array([1,0,1])
# x2_dt = np.array([[1,1,0],[- 0.4,0,1],[- 0.1,3,0]])
# D = np.array([0,0.4])
# dt = 0.0075
# T_end = 15
# M1 = 10
# M2 = 15
# N1 = 300
# N2 = 300
# mu = np.array([0,10])
# sigma = 1 / 2 * np.eye(2)
# #}
# #{
# van der pol oscillator
x1_dt = np.array([1,0,1])
x2_dt = np.array([[- 0.1,2,1],[0.1,0,1],[- 1,1,0]])
D = np.array([0,0.5])
dt = 0.01
T_end = 50
M1 = 10
M2 = 10
N1 = 200
N2 = 200
mu = np.array([4,4])
sigma = 1 / 2 * np.eye(2)
#}

D1,D2,D11,D22,x1,x2 = make_matrix(M1,N1,M2,N2,x1_dt,x2_dt,D)
t,p = solve_advDif(D1,D2,D11,D22,x1,x2,dt,T_end,mu,sigma)