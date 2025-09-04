import numpy as np
import math

##################################################################################################
##################################################################################################
##################################################################################################
#################################### Inverse Laplace Transform ###################################
##################################################################################################
##################################################################################################
##################################################################################################
# de hoog method
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

##################################################################################################
##################################################################################################
##################################################################################################
############################# Single-porosity & dual-porosity models #############################
##################################################################################################
##################################################################################################
##################################################################################################
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

def QE_method(t, X, dT, T, cin, param, setup, a, NSUM, accelerate):
     pi = math.pi
     T0 = T/2

     xa = base_fun(X, a*np.ones([2,1]), dT, cin, param, setup)
     y0 =  (0.5*xa[0,:].real)

     k = np.arange(pi/T0, pi/T0 * (NSUM+1), pi/T0).reshape(-1,1)
     y1 = base_fun(X, a+1j*k, dT, cin, param, setup) 

     y = np.concatenate((y0.reshape(1,1), y1), axis = 0)
     return np.array([qe(t, T, y, a, NSUM, accelerate)])
 
##################################################################################################
##################################################################################################
##################################################################################################
########################## Dual-permeability & triple-porosity models ############################
##################################################################################################
##################################################################################################
##################################################################################################
# get concentration in the time domain
def fcfun0(X, gsi, c0, rs):# Define a symbolic variable
    fc = np.array(rs)*0.0
    return fc

# get coefficients from inverse Laplace 
def fcfun(X, gsi, c0, rs):# Define a symbolic variable
    fc = np.array(rs)*0.0
    return fc

def base_fun_TP(s, X, dT, cin, param, setup):
    # get all functional values with respect to s
    ## hf & hm function in the first advective domain
    [Rf,  Pe_f, mu_f, Kp_fm, 
     R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
     Rm,  Pe_m, mu_m, Kp_mf, 
     R_sw_m, A_sw_m, B_sw_m, R_aw_m,A_aw_m, B_aw_m, Kp_mim,
     Rim, Kp_ims, mu_ims, 
     R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims] = param
    
    Hf  = s*Rf + Kp_fm + mu_f
    if setup['f_SWI_sites'] !=  0: 
        Hf += np.sum((s*R_sw_f*A_sw_f)/(s+B_sw_f), axis = 1).reshape(-1,1)         
    if setup['f_AWI_sites'] !=  0: 
        Hf += np.sum((s*R_aw_f*A_aw_f)/(s+B_aw_f), axis = 1).reshape(-1,1)

    Hm  = s*Rm + Kp_mf + mu_m
    if setup['m_SWI_sites'] !=  0: 
        Hm += np.sum((s*R_sw_m*A_sw_m)/(s+B_sw_m), axis = 1).reshape(-1,1)         
    if setup['m_AWI_sites'] !=  0: 
        Hm += np.sum((s*R_aw_m*A_aw_m)/(s+B_aw_m), axis = 1).reshape(-1,1)
        
    gsi = 0.0*(s*Rim + Kp_ims + mu_ims)
    if setup['perm_domain'] == 3:
        gsi = s*Rim + Kp_ims + mu_ims
        if setup['m_SWI_sites'] !=  0: 
            gsi += s*(R_sw_ims*A_sw_ims)/(s+B_sw_ims) 
        if setup['m_AWI_sites'] !=  0:
            gsi += s*(R_aw_ims*A_aw_ims)/(s+B_aw_ims)    
        Hm += np.sum( Kp_mim * (gsi-Kp_ims) / gsi , axis = 1).reshape(-1,1) 
        
    P1, P2 = Pe_f, Pe_m
    hs1, hs2 = Hf, Hm
    k1, k2 = Kp_fm, -Kp_mf
    ## get the roots for the polynomial: f(r) = (r-r11)(r-r12)(r-r21)(r-r22)
    r11 = 0.5*P1 - 0.5*P1*np.sqrt(1+4*hs1/P1)
    r12 = 0.5*P1 + 0.5*P1*np.sqrt(1+4*hs1/P1)
    r21 = 0.5*P2 - 0.5*P2*np.sqrt(1+4*hs2/P2)
    r22 = 0.5*P2 + 0.5*P2*np.sqrt(1+4*hs2/P2)
    r0s = np.concatenate( (r11, r12, r21, r22), axis=1)
    '''print('inital roots are ', r0s)'''
    ## sovle the roots for the polynomial: f(r) = (r-r11)(r-r12)(r-r21)(r-r22) + P1*P2*k1*k2
    coefs = np.concatenate( (np.ones((r0s.shape[0],1)), 
                             -(r11 + r12 + r21 + r22), 
                             r11*r12+r21*r22+(r11+r12)*(r21+r22), 
                             -(r12*r21*r22+r11*r21*r22+r11*r12*r22+r11*r12*r21), 
                             r11*r12*r21*r22+P1*P2*k1*k2), axis=1)


    rs_list = []
    for coef in coefs:
        roots = np.roots(coef)
        rs_list.extend(roots)
    rs_col = np.array(rs_list)
    rs = rs_col.reshape(coefs.shape[0], coefs.shape[1]-1)

    ## solve for A, B, C, D, E, F
    r1 = np.array(rs[:,0].reshape(-1,1))
    r2 = np.array(rs[:,1].reshape(-1,1))
    r3 = np.array(rs[:,2].reshape(-1,1))
    r4 = np.array(rs[:,3].reshape(-1,1))
    jac0 = np.concatenate( ( np.ones((r0s.shape[0],4)), 
                             -(r2+r3+r4), 
                             -(r1+r3+r4), 
                             -(r1+r2+r4), 
                             -(r1+r2+r3),
                             r2*r3+r2*r4+r3*r4, 
                             r1*r3+r1*r4+r3*r4, 
                             r1*r2+r1*r4+r2*r4, 
                             r1*r2+r1*r3+r2*r3,
                             -r2*r3*r4, 
                             -r1*r3*r4, 
                             -r1*r2*r4, 
                             -r1*r2*r3), 
                             axis=1)

    jac = jac0.reshape(r0s.shape[0], 4, 4)

    ### for A
    b = np.concatenate( ( np.zeros((r0s.shape[0],1)), np.ones((r0s.shape[0],1)), -(r21+r22), r21*r22 ), axis=1)    
    A = np.linalg.solve(jac, b)
    ### for B
    b = np.concatenate( ( np.ones((r0s.shape[0],1)), -(r21+r22), r21*r22, np.zeros((r0s.shape[0],1)) ), axis=1)      
    B = np.linalg.solve(jac, b)
    ### for C
    b = np.concatenate( ( np.zeros((r0s.shape[0],1)), np.zeros((r0s.shape[0],1)), np.zeros((r0s.shape[0],1)), 1.0*np.ones((r0s.shape[0],1)) ), axis=1)    
    C = np.linalg.solve(jac, b)
    ### for D
    b = np.concatenate( ( np.zeros((r0s.shape[0],1)), np.zeros((r0s.shape[0],1)), 1.0*np.ones((r0s.shape[0],1)), np.zeros((r0s.shape[0],1)) ), axis=1)   
    D = np.linalg.solve(jac, b)
    ### for E
    b = np.concatenate( ( np.zeros((r0s.shape[0],1)), np.ones((r0s.shape[0],1)), -(r11+r12), r11*r12 ), axis=1) 
    E = np.linalg.solve(jac, b)
    ### for F
    b = np.concatenate( ( np.ones((r0s.shape[0],1)), -(r11+r12), r11*r12, np.zeros((r0s.shape[0],1)) ), axis=1)
    F = np.linalg.solve(jac, b)


    ## get coefficients for solving boundary concentration
    gs_adv = np.ones(s.shape)
    gs1i_c1i0_inf = fcfun(1e10, gsi, 0, rs)
    gs2i_c2i0_inf = fcfun(1e10, gsi, 0, rs)
    f1inf = -A/P2* (Rf*fcfun(1e10, gs_adv, 0, rs)+gs1i_c1i0_inf) + C*k1* (Rm*fcfun(1e10, gs_adv, 0, rs)+gs2i_c2i0_inf)
    f2inf = -E/P1* (Rm*fcfun(1e10, gs_adv, 0, rs)+gs2i_c2i0_inf) - C*k2* (Rf*fcfun(1e10, gs_adv, 0, rs)+gs1i_c1i0_inf)

    ## final parameters
    alpha1 = A/P2 - C*k1
    alpha2 = E/P1 + C*k2
    beta1 = B/P1/P2
    beta2 = F/P1/P2
    gamma1 = D*k1/P2
    gamma2 = D*k2/P1

    c1_in = cin
    c2_in = cin
    
    ## solve boundary concentration
    LTc1_in = c1_in / s * (1-np.exp(-dT*s))
    LTc2_in = c2_in / s * (1-np.exp(-dT*s))

    ### Find the indices and entry of the maximum real part in each row
    rmax_id = np.argmax(rs.real, axis=1)
    rmax = rs[np.arange(rs.shape[0]), rmax_id].reshape(-1,1)
    '''print('roots are ', rs)'''
    r_01 = np.array(rs)
    r_01[r_01.real <= 0] = 0
    r_01[r_01.real > 0] = 1
    r_diff = rs - rmax

    ### solve boundary concentration
    fa = np.sum(r_01*(np.exp(r_diff*X)*beta1), axis=1).reshape(-1,1)
    fb = np.sum(r_01*(np.exp(r_diff*X)*gamma1), axis=1).reshape(-1,1)
    fc = np.sum(r_01*(np.exp(r_diff*X)*gamma2), axis=1).reshape(-1,1)
    fd = np.sum(r_01*(np.exp(r_diff*X)*beta2), axis=1).reshape(-1,1)
    res1 = np.sum(r_01*(np.exp(r_diff*X)*(-f1inf + alpha1*LTc1_in)), axis=1).reshape(-1,1)
    res2 = np.sum(r_01*(np.exp(r_diff*X)*(-f2inf + alpha2*LTc2_in)), axis=1).reshape(-1,1)

    fa_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*beta1), axis=1).reshape(-1,1)
    fb_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*gamma1), axis=1).reshape(-1,1)
    fc_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*gamma2), axis=1).reshape(-1,1)
    fd_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*beta2), axis=1).reshape(-1,1)
    res1_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*(-f1inf + alpha1*LTc1_in)), axis=1).reshape(-1,1)
    res2_grad = np.sum(r_01*(r_diff*np.exp(r_diff*X)*(-f2inf + alpha2*LTc2_in)), axis=1).reshape(-1,1)

    cb0 = (fd*res1+fb*res2)/(fa*fd+fb*fc)
    cb1 = (-fc*res1+fa*res2)/(fa*fd+fb*fc)
    cb = np.concatenate((cb0,cb1),axis=1)

    cb0_grad = ( (fd_grad*res1+fd*res1_grad+fb_grad*res2+fb*res2_grad)*(fa*fd+fb*fc) - (fd*res1+fb*res2)*(fa_grad*fd+fa*fd_grad+fb_grad*fc+fb*fc_grad) )/(fa*fd+fb*fc)/(fa*fd+fb*fc)    
    cb1_grad = ( (-fc_grad*res1-fc*res1_grad+fa_grad*res2+fa*res2_grad)*(fa*fd+fb*fc)- (-fc*res1+fa*res2)*(fa_grad*fd+fa*fd_grad+fb_grad*fc+fb*fc_grad) )/(fa*fd+fb*fc)/(fa*fd+fb*fc)
    cb_grad = np.concatenate((cb0_grad,cb1_grad),axis=1)


    ## solve concentrationat s & X
    gs1i_c1i0_X = fcfun(X, gsi, 0, rs)
    gs2i_c2i0_X = fcfun(X, gsi, 0, rs)

    f1X = -A/P2* (Rf*fcfun(X, gs_adv, 0, rs)+gs1i_c1i0_X) + C*k1* (Rm*fcfun(X, gs_adv, 0, rs)+gs2i_c2i0_X)
    f2X = -E/P1* (Rm*fcfun(X, gs_adv, 0, rs)+gs2i_c2i0_X) - C*k2* (Rf*fcfun(X, gs_adv, 0, rs)+gs1i_c1i0_X)


    gs1i_c1i0_X0 = fcfun0(X, gsi, 0, rs)
    gs2i_c2i0_X0 = fcfun0(X, gsi, 0, rs)
    f1X_grad = -A/P2* (Rf*fcfun0(X, gs_adv, 0, rs)+gs1i_c1i0_X0) + C*k1* (Rm*fcfun0(X, gs_adv, 0, rs)+gs2i_c2i0_X0)
    f2X_grad = -E/P1* (Rm*fcfun0(X, gs_adv, 0, rs)+gs2i_c2i0_X0) - C*k2* (Rf*fcfun0(X, gs_adv, 0, rs)+gs1i_c1i0_X0)


    # volume-based concentration
    c1 = P1*P2*np.sum( np.exp(rs*X)*(f1X-alpha1*LTc1_in+beta1*cb[:,0].reshape(-1,1)-gamma1*cb[:,1].reshape(-1,1)) , axis=1).reshape(-1,1)
    c2 = P1*P2*np.sum( np.exp(rs*X)*(f2X-alpha2*LTc2_in+beta2*cb[:,1].reshape(-1,1)+gamma2*cb[:,0].reshape(-1,1)) , axis=1).reshape(-1,1)
    # flux-based concentration
    c1 = P2*np.sum( np.exp(rs*X)
                   * ( (P1-rs)*(f1X-alpha1*LTc1_in+beta1*cb[:,0].reshape(-1,1)-gamma1*cb[:,1].reshape(-1,1)) 
                   - (f1X_grad+beta1*cb_grad[:,0].reshape(-1,1)-gamma1*cb_grad[:,1].reshape(-1,1)) )
                   , axis=1 ).reshape(-1,1)

    c2 = P1*np.sum( np.exp(rs*X) 
                   * ( (P2-rs)*(f2X-alpha2*LTc2_in+beta2*cb[:,1].reshape(-1,1)+gamma2*cb[:,0].reshape(-1,1)) 
                   - (f2X_grad+beta2*cb_grad[:,1].reshape(-1,1)+gamma2*cb_grad[:,0].reshape(-1,1)) )
                   , axis=1 ).reshape(-1,1)
    return np.concatenate((c1, c2), axis=1)


def QE_method_TP(t, X, dT, T, cin, param, setup, a, NSUM, accelerate):
    
    [Rf,  Pe_f, mu_f, Kp_fm, 
     R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
     Rm,  Pe_m, mu_m, Kp_mf, 
     R_sw_m, A_sw_m, B_sw_m, R_aw_m,A_aw_m, B_aw_m, Kp_mim,
     Rim, Kp_ims, mu_ims, 
     R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims] = param
    
    pi = math.pi
    T0 = T/2
    c0 = 0.5*base_fun_TP(a * np.ones((2,1)), X, dT, cin, param, setup)[0,:].real
    ain = a + 1j * np.arange(pi/T0, (NSUM+1) * pi/T0, pi/T0).reshape(-1,1)
    y0 = base_fun_TP(ain, X, dT, cin, param, setup)
    y = np.vstack((c0, y0))
    # print(y.shape)
    return np.array([qe(t, T, y[:,0], a, NSUM, accelerate), qe(t, T, y[:,1], a, NSUM, accelerate)])
