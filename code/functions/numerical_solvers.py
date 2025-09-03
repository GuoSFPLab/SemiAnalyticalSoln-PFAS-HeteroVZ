import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

##################################################################################################
##################################################################################################
##################################################################################################
############################# Single-porosity & dual-porosity models #############################
##################################################################################################
##################################################################################################
##################################################################################################
def num_SinglePoroDualPoro(X, dT, T, dt, cin, param, thetas, setup):
    
    [Rf,  Pe_f, mu_f, Kp_fim, 
     R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
     Rim, Kp_ims, mu_ims, 
     R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims] = param
    
    [wf, theta_f, wim, theta_im, t_scalar] = thetas
    
    Nim  = int(setup['ImZones'])
    dx = X/setup['dx_num']
    nx = int(X/dx)
    N = nx
    # initial conditions
    c0 = np.zeros(N) # initial concentration in the advective domain
    c = np.zeros(N)
    c0_f_aw = np.zeros((N, A_aw_f.shape[0])) # initial concentration at the air-water interfaces
    c0_f_sw = np.zeros((N, A_sw_f.shape[0])) # initial concentration at the solid-water interfaces
    c0_ims = np.zeros((N, Nim))
    c0_im_aw = np.zeros((N, Nim)) # initial concentration at the air-water interfaces
    c0_im_sw = np.zeros((N, Nim)) # initial concentration at the solid-water interfaces
    
    # build Jacobian
    Jac = np.zeros(N**2)
    idm = np.arange(N,N*(N-1),N) + np.arange(1,N-1,1)
    
    # Fracture domain
    Jac[idm] = 2/Pe_f/dx/dx-1/dx + Rf/dt + mu_f # at x
    if setup['f_SWI_sites'] !=  0: 
        Jac[idm] += (R_sw_f*A_sw_f/(1+dt*B_sw_f)).sum()
    if setup['f_AWI_sites'] !=  0: 
        Jac[idm] += (R_aw_f*A_aw_f/(1+dt*B_aw_f)).sum()    
    Jac[idm - 1] = (-1/Pe_f/dx/dx) # X - dx
    Jac[idm + 1] = (-1/Pe_f/dx/dx+1/dx) # X + dx
        
    # Immobile matrix domain
    if wim > 0 and wim < 1: 
        Sims = Rim/dt + Kp_ims + mu_ims 
        if setup['f_SWI_sites'] !=  0: 
            Sims += R_sw_ims * A_sw_ims / (1+dt*B_sw_ims) 
        if setup['f_AWI_sites'] !=  0: 
            Sims += R_aw_ims * A_aw_ims / (1+dt*B_aw_ims)   
    if wim > 0 and wim < 1:
        Jac[idm] += (Kp_fim * (Sims - Kp_ims) / Sims).sum()
        
    # Boundaries
    Jac[0], Jac[1] = (1+1/Pe_f/dx), (-1/Pe_f/dx) # inlet
    Jac[(N+1)*(N-1)], Jac[(N+1)*(N-1)-1] = -1, 1 # outlet
    Jacm = Jac.reshape(N, N)
    Jacm_sparse = csr_matrix(Jacm)
    
    # time marching
    time, conc = [], [] 
    time.append(0)
    conc.append(c[-1]/cin)
    nt = 0
    for t in np.arange(0.0, T+dt, dt):
        if nt%int(T/dt/10) == 0 or t > T-dt:
            print(f"current time step is {t:.2f} [PV], total time is {T:.2f} [PV]!")
        nt += 1
        # right hand side
        b = Rf/dt * c0
        if setup['f_SWI_sites'] !=  0: 
            b += ( c0_f_sw*((R_sw_f*B_sw_f)/(1+dt*B_sw_f)) ).sum(axis=1) 
        if setup['f_AWI_sites'] !=  0:
            b+= ( c0_f_aw*((R_aw_f*B_aw_f)/(1+dt*B_aw_f)) ).sum(axis=1)
            
        if wim > 0 and wim < 1:
            b += ( (c0_ims*(Rim/dt)) * (Kp_fim/Sims) ).sum(axis=1)
            if setup['f_SWI_sites'] !=  0: 
                b += ( (c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims))) * (Kp_fim/Sims) ).sum(axis=1) 
            if setup['f_AWI_sites'] !=  0: 
                b += ( (c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims))) * (Kp_fim/Sims) ).sum(axis=1)
            
        # boundary condition
        b[0] = cin if t <= dT else 0 
        b[N-1] = 0
        # solve
        c = spsolve(Jacm_sparse, b)
        # update previous time step
        c0 = c
        if setup['f_SWI_sites'] !=  0: 
            c0_f_sw = c0_f_sw/(1+dt*B_sw_f) + c[:, np.newaxis]*((dt*A_sw_f)/(1+dt*B_sw_f))
        if setup['f_AWI_sites'] !=  0: 
            c0_f_aw = c0_f_aw/(1+dt*B_aw_f) + c[:, np.newaxis]*((dt*A_aw_f)/(1+dt*B_aw_f))
            
        if wim > 0 and wim < 1: # Add immobile zone
            c0_ims = ( c[:, np.newaxis]*Kp_ims + c0_ims*(Rim/dt) ) / Sims  
            if setup['f_SWI_sites'] !=  0: 
                c0_ims += ( c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims)) ) / Sims
                c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims)) # update kenitic SWI conc
            if setup['f_AWI_sites'] !=  0:    
                c0_ims += ( c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims)) ) / Sims
                #c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims)) # update kenitic SWI conc
                c0_im_aw = c0_im_aw*(1/(1+dt*B_aw_ims)) + c0_ims*((dt*A_aw_ims)/(1+dt*B_aw_ims)) # update kenitic AWI conc
        # store results
        time.append(t * t_scalar)
        conc.append(c[-1]/cin)
    return time, conc


##################################################################################################
##################################################################################################
##################################################################################################
########################## Dual-permeability & triple-porosity models ############################
##################################################################################################
##################################################################################################
##################################################################################################
def num_DualPermTriPoro(X, dT, T, dt, cin, param, thetas, setup):
    
    Nim  = int(setup['ImZones'])
    
    [Rf,  Pe_f, mu_f, Kp_fm, 
     R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
     Rm,  Pe_m, mu_m, Kp_mf, 
     R_sw_m, A_sw_m, B_sw_m, R_aw_m,A_aw_m, B_aw_m, Kp_mim,
     Rim, Kp_ims, mu_ims, 
     R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims] = param
    
    [wf, theta_f, wm, theta_m, wim, theta_im, t_scalar] = thetas
    
    dx =  X/setup['dx_num']
    nx = int(X/dx)
    N = nx
    # initial conditions
    cf0 = np.zeros(N) # initial concentration in the advective domain
    cf = np.zeros(N)
    c0_f_aw = np.zeros((N, A_aw_f.shape[0])) # initial concentration at the air-water interfaces
    c0_f_sw = np.zeros((N,A_sw_f.shape[0])) # initial concentration at the solid-water interfaces
    cm0 = np.zeros(N) # initial concentration in the advective domain
    cm = np.zeros(N)
    c0_m_aw = np.zeros((N, A_aw_m.shape[0])) # initial concentration at the air-water interfaces
    c0_m_sw = np.zeros((N, A_sw_m.shape[0])) # initial concentration at the solid-water interfaces
    c0_ims = np.zeros((N, Nim))
    c0_im_aw = np.zeros((N, Nim)) # initial concentration at the air-water interfaces
    c0_im_sw = np.zeros((N, Nim)) # initial concentration at the solid-water interfaces
    
    # build Jacobian
    Jac = np.zeros((2*N)**2)
    idf = np.arange(2, 2*N-2, 2) * (2*N) + np.arange(2, 2*N-2, 2)
    idm = np.arange(3, 2*N-2, 2) * (2*N) + np.arange(3, 2*N-2, 2)
    
    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(precision=1, suppress=True)
    # Fracture domain
    Jac[idf] = 2/Pe_f/dx/dx-1/dx + Rf/dt + mu_f + Kp_fm# at x
    Jac[idf + 1] += -Kp_fm
    if setup['f_SWI_sites'] !=  0: 
        Jac[idf] += (R_sw_f*A_sw_f/(1+dt*B_sw_f)).sum()
    if setup['f_AWI_sites'] !=  0: 
        Jac[idf] += (R_aw_f*A_aw_f/(1+dt*B_aw_f)).sum()    
    
    Jac[idf - 2] = (-1/Pe_f/dx/dx) # X - dx
    Jac[idf + 2] = (-1/Pe_f/dx/dx+1/dx) # X + dx
    
    
    # Matrix domain
    Jac[idm] = 2/Pe_m/dx/dx-1/dx + Rm/dt + mu_m + Kp_mf # at x
    Jac[idm - 1] += -Kp_mf  
    if setup['m_SWI_sites'] !=  0: 
        Jac[idm] += (R_sw_m*A_sw_m/(1+dt*B_sw_m)).sum()
    if setup['m_AWI_sites'] !=  0: 
        Jac[idm] += (R_aw_m*A_aw_m/(1+dt*B_aw_m)).sum()    
    Jac[idm - 2] = (-1/Pe_m/dx/dx) # X - dx
    Jac[idm + 2] = (-1/Pe_m/dx/dx+1/dx) # X + dx
        
    
    # Immobile matrix domain
    if wim > 0 and wim < 1: 
        Sims = Rim/dt + Kp_ims + mu_ims 
        if setup['f_SWI_sites'] !=  0: 
            Sims += R_sw_ims * A_sw_ims / (1+dt*B_sw_ims) 
        if setup['f_AWI_sites'] !=  0: 
            Sims += R_aw_ims * A_aw_ims / (1+dt*B_aw_ims)   
    if wim > 0 and wim < 1:
        Jac[idm] += (Kp_mim * (Sims - Kp_ims) / Sims).sum()
       
    #print(Jac.reshape(2*N, 2*N)) 
    
    # Boundaries
    Jac[0], Jac[2] = (1+1/Pe_f/dx), (-1/Pe_f/dx) # inlet
    Jac[(2*N)*(2*N-1)-2], Jac[(2*N)*(2*N-1)-4] = -1, 1 # outlet
    
    Jac[2*N+1], Jac[2*N+3] = (1+1/Pe_m/dx), (-1/Pe_m/dx) # inlet
    Jac[(2*N+1)*(2*N-1)], Jac[(2*N+1)*(2*N-1)-2] = -1, 1 # outlet
    
    #print(Jac.reshape(2*N, 2*N)) 
    
    Jacm = Jac.reshape(2*N, 2*N)
    Jacm_sparse = csr_matrix(Jacm)
    # time marching
    time, concf, concm, conc_ave = [], [], [], [] 
    time.append(0)
    concf.append(cf[-1]/cin)
    concm.append(cm[-1]/cin)
    conc_ave.append((wf*theta_f*cf[-1] + wm*theta_m*cm[-1])/(wf*theta_f+wm*theta_m)/cin)
    nt = 0
    for t in np.arange(0.0, T+dt, dt):
        if nt%int(T/dt/10) == 0 or t > T-dt:
            print(f"current time step is {t:.2f} [PV], total time is {T:.2f} [PV]!")
        nt += 1
        # right hand side
        bf = Rf/dt * cf0
        if setup['f_SWI_sites'] !=  0: 
            bf += ( c0_f_sw*((R_sw_f*B_sw_f)/(1+dt*B_sw_f)) ).sum(axis=1) 
        if setup['f_AWI_sites'] !=  0:
            bf += ( c0_f_aw*((R_aw_f*B_aw_f)/(1+dt*B_aw_f)) ).sum(axis=1)
            
        
        bm = Rm/dt * cm0
        if setup['m_SWI_sites'] !=  0: 
            bm += ( c0_m_sw*((R_sw_m*B_sw_m)/(1+dt*B_sw_m)) ).sum(axis=1) 
        if setup['m_AWI_sites'] !=  0:
            bm += ( c0_m_aw*((R_aw_m*B_aw_m)/(1+dt*B_aw_m)) ).sum(axis=1)
            
        if wim > 0 and wim < 1:
            bm += ( (c0_ims*(Rim/dt)) * (Kp_mim/Sims) ).sum(axis=1)
            if setup['f_SWI_sites'] !=  0: 
                bm += ( (c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims))) * (Kp_mim/Sims) ).sum(axis=1) 
            if setup['f_AWI_sites'] !=  0: 
                bm += ( (c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims))) * (Kp_mim/Sims) ).sum(axis=1)
            
        # boundary condition
        bf[0] = cin if t <= dT else 0 
        bf[N-1] = 0
        bm[0] = cin if t <= dT else 0 
        bm[N-1] = 0
        
        b = np.column_stack((bf, bm)).ravel()
        
        # solve
        c = spsolve(Jacm_sparse, b)
        cf = c[0::2]
        cm = c[1::2]
        
        # update previous time step
        # Fracture
        cf0 = cf
        if setup['f_SWI_sites'] !=  0: 
            c0_f_sw = c0_f_sw/(1+dt*B_sw_f) + cf[:, np.newaxis]*((dt*A_sw_f)/(1+dt*B_sw_f))
        if setup['f_AWI_sites'] !=  0: 
            c0_f_aw = c0_f_aw/(1+dt*B_aw_f) + cf[:, np.newaxis]*((dt*A_aw_f)/(1+dt*B_aw_f))
            
        # Mobile matrix 
        cm0 = cm
        if setup['m_SWI_sites'] !=  0: 
            c0_m_sw = c0_m_sw/(1+dt*B_sw_m) + cm[:, np.newaxis]*((dt*A_sw_m)/(1+dt*B_sw_m))
        if setup['m_AWI_sites'] !=  0: 
            c0_m_aw = c0_m_aw/(1+dt*B_aw_m) + cm[:, np.newaxis]*((dt*A_aw_m)/(1+dt*B_aw_m))
         
        # Immobile matrix 
        if wim > 0 and wim < 1: # Add immobile zone
            c0_ims = ( cm[:, np.newaxis]*Kp_ims + c0_ims*(Rim/dt) ) / Sims  
            if setup['m_SWI_sites'] !=  0: 
                c0_ims += ( c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims)) ) / Sims
                c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims)) # update kenitic SWI conc
            if setup['m_AWI_sites'] !=  0:    
                c0_ims += ( c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims)) ) / Sims
                c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims))
                c0_im_aw = c0_im_aw*(1/(1+dt*B_aw_ims)) + c0_ims*((dt*A_aw_ims)/(1+dt*B_aw_ims)) # update kenitic AWI conc
        # store results
        time.append(t * t_scalar)
        concf.append(cf[-1]/cin)
        concm.append(cm[-1]/cin)
        conc_ave.append((wf*theta_f*cf[-1] + wm*theta_m*cm[-1])/(wf*theta_f+wm*theta_m)/cin)
    return time, concf, concm, conc_ave