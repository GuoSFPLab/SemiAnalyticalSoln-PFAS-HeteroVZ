import numpy as np
from scipy.stats import lognorm
import math
import pandas as pd
import sympy as sp

# Base function for the MIM model in a finite domain
def base_fun(X, s, dT, cin, param, setup):
    
    [Rf,  Pe_f, mu_f, Kp_fim, R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
     Rim, Kp_ims, mu_ims, R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims] = param
    
    H  = s*Rf + mu_f
    if setup['perm_domain'] == 1:
        gsi = s*Rim + Kp_ims + mu_ims
        if setup['f_SWI_sites'] !=  0: 
            gsi += s*(R_sw_ims*A_sw_ims)/(s+B_sw_ims) 
        if setup['f_AWI_sites'] !=  0:
            gsi += s*(R_aw_ims*A_aw_ims)/(s+B_aw_ims)    
        H += np.sum( Kp_fim * (gsi-Kp_ims) / gsi , axis = 1).reshape(-1,1)
        
    if setup['f_SWI_sites'] !=  0: 
        H += np.sum((s*R_sw_f*A_sw_f)/(s+B_sw_f), axis = 1).reshape(-1,1)
        
    if setup['f_AWI_sites'] !=  0: 
        H += np.sum((s*R_aw_f*A_aw_f)/(s+B_aw_f), axis = 1).reshape(-1,1)

    A = Pe_f/2*(1-np.sqrt(1+4*H/Pe_f))
    B = Pe_f/2*(1+np.sqrt(1+4*H/Pe_f))
    
    cin_tau = cin / s * (1-np.exp(-dT*s))
    # volume-average concentration
    #yb = Pe_f/B*cin_tau*np.exp(X*A)
    # flux-average concentration
    yb = (Pe_f-A)/B*cin_tau*np.exp(X*A)
    return yb

# de hoog method
def QE_method(t, X, dT, T, cin, param, setup, a, NSUM, accelerate):
     pi = math.pi
     T0 = T/2

     xa = base_fun(X, a*np.ones([2,1]), dT, cin, param, setup)
     y0 =  (0.5*xa[0,:].real)

     k = np.arange(pi/T0, pi/T0 * (NSUM+1), pi/T0).reshape(-1,1)
     y1 = base_fun(X, a+1j*k, dT, cin, param, setup) 

     y = np.concatenate((y0.reshape(1,1), y1), axis = 0)
     return np.array([qe(t, T, y, a, NSUM, accelerate)])
 

def qe(t, T, A, a, NSUM, accelerate):
    QE = np.zeros((NSUM+1,NSUM+1), dtype = complex)
    # col 0, 2, 4 ... -> e_0, e_1, e_2, 
    # col 1, 3, ... -> q_1, q_2 
    # row 0, 1, 2, 3, 4, ... -> e^0, e^1, e^2, e^3, e^4
    # row 0, 1, 2, 3, ... -> q^0, q^1, q^2, q^3
    # e^i_j = QE[i,j*2]
    # q^i_j = QE[i,j*2-1] j >= 1
    # e^i_0 = 0
    # q^i_1 = a_i+1/a_i
    pi = math.pi
    T0 = T/2
    z = np.exp(1j*pi*t/T0)

    for ir in range(NSUM):
        QE[ir,1] = A[ir+1]/A[ir] 

    # compute e & q
    for ic in range(2,NSUM+1,1):
        ir = np.arange(NSUM+1-ic)
        if ic%2 == 0:
            # e_r^i = q^i+1_r - q^i_r + e^i+1_r-1
            QE[ir,ic] = QE[ir+1,ic-1] - QE[ir,ic-1] + QE[ir+1,ic-2]
            # print(ir,ic)
        else:
            # q_r^i = q^i+1_r-1 * e^i+1_r-1 / e^i_r-1
            QE[ir,ic] = QE[ir+1,ic-1]/QE[ir,ic-1]*QE[ir+1,ic-2]

    #print(QE.real)       
    D = -QE[0,]
    D[0] = A[0]
    A0 = np.zeros((NSUM+2), dtype = complex)
    B0 = np.zeros((NSUM+2), dtype = complex)
    A0[0] = 0
    A0[1] = D[0]
    B0[0] = 1
    B0[1] = 1
    for n in range(2,NSUM+2,1):
        # print(n)
        A0[n] = A0[n-1]+D[n-1]*z*A0[n-2]
        B0[n] = B0[n-1]+D[n-1]*z*B0[n-2]

    y = A0[-1]/B0[-1]
    h2M = 0.5*(1+(D[-2]-D[-1])*z)
    R2M = -h2M*(1-np.sqrt(1+D[-1]*z/h2M/h2M))
    if accelerate == 1:
        y = (A0[-2]+R2M*A0[-3])/(B0[-2]+R2M*B0[-3])

    return np.exp(a*t)/T0*y.real