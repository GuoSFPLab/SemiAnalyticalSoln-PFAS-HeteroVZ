import numpy as np
from scipy.stats import lognorm
import math
import pandas as pd
import sympy as sp
from .inverse_laplace import QE_method

def Aaw_func_thermo(sigma0,poro,alpha,n,th,thr,ths,sf):
    #Computing air-water interfacial area using the thermodynamic approach
    # Aaw       air-water interfacial area (cm^2/cm^3)
    # sigma0    surface tension (dyn/cm)
    # poro      porosity (-)
    # alphal    V-G parameter (cm^-1)
    # n         V-G parameter (-)
    # th        water content (-)
    # thr       residual water content (-)
    # ths       saturated water content (-)
    # rhow      water density (kg/m^3)
    # g         gravity acceleration (m/s^2)
    # Pc        capillary pressure
    # sf        scaling factor to correct the thermodynamic-based Aaw (-)
    rhow = 1000
    g = 9.81
    m = 1 - 1/n
    Sr = thr/ths
    Sw = np.linspace(th/ths,1,1000)
    Pc = lambda Sw: (((1-Sr)/(Sw-Sr))**(1/m) - 1)**(1/n)/alpha/100*rhow*g
    Aaw = 10*np.trapz(poro/sigma0*Pc(Sw),Sw)
    Aaw = Aaw*sf
    return Aaw  

def Aaw_func_tracer(sw,x2,x1,x0):
    #Computing air-water interfacial area using the thermodynamic approach    
    # Aaw           air-water interfacial area (cm^2/cm^3)
    # sw            water saturation
    # x2, x1, x0    polynomial fitting coefficients
    Aaw = x2*sw**2 + x1*sw + x0
    return Aaw


def lognormal_pdf(x, mu, sigma):
        """
        - x: The sample value
        - mu: Mean of the natural logarithm of the distribution
        - sigma: Standard deviation of the natural logarithm of the distribution
        """
        pdf_value = 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-((np.log(x) - mu)**2) / (2 * sigma**2))
        return pdf_value
    
def find_x_at_given_f(mu, sigma, loc):  
    fmu = lognormal_pdf(np.exp(mu), mu, sigma) # frequency at x = np.exp(mu)
    ref = 1.e-2*fmu
    if loc == 0: # compute lower bound
        lv = mu
        uv = mu
        while lognormal_pdf(np.exp(lv), mu, sigma) > ref:
            lv -= abs(mu)     
        while abs(lognormal_pdf(np.exp((lv+uv)/2), mu, sigma)-ref)/ref > 1.e-2:
            if lognormal_pdf(np.exp((lv+uv)/2), mu, sigma) > ref:
                uv = (lv+uv)/2
            else:
                lv = (lv+uv)/2  
        return (lv+uv)/2
    else: # compute upper bound
        lv = mu
        uv = mu
        while lognormal_pdf(np.exp(uv), mu, sigma) > ref:
            uv += abs(mu)
        return uv


def analytical_solution(setup, soil, pfas):
    
    # Parameters
    hour = 3600
    day = 24*hour
    year = 365*day
    sigma0 = 71       # dyn/cm
    Rg = 8.31         # gas constant [J/K/mol]
    Tp = 293.15       # temperature [K]
    
    M0 = pfas['Mw']      # g/mol
    I0 = setup['I0']/365 # cm/year to cm/day
    L =  setup['L']      # cm
    cin, c_rep = pfas['cin'], pfas['c_rep']   # mg/L
    a, b = pfas['a'], pfas['b']
    Kaw = setup['AWI_type'] * 0.1*M0*sigma0*b/(Rg*Tp*(a+c_rep)) # [cm]    
    D0  = pfas['D0'] #cm^2/s
    
    ## read fracture data
    wf = soil['f_w']
    n_vg_f = soil['f_n_VG']  # VG parameter
    m_vg_f = 1 - 1/n_vg_f    # VG parameter 
    alpha_VG_f = soil['f_alpha_VG']/100 # VG parameter 
    alphaL_f = soil['f_alpha_L']
    poro_f = soil['f_theta_s']   # [-] 
    rhob_f = soil['f_rhob']   # g/cm^3
    tau_f =  soil['f_tau']
    kd_f = pfas['f_kd']    # cm^3/g
    if kd_f == 0:
        kf_f, N_f = pfas['f_kf'], pfas['f_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_f = kf_f * np.power(c_rep,N_f-1) # L/kg -> cm^3/g  
    kd_f *= setup['SWI_type']
    mu_f  = pfas['f_mu']/hour
    Faw_f = 1 if setup['AWI_sites'] == 0 else pfas['f_F_aw']
    k_aw_f = pfas['f_k_aw_ave']
    Fsw_f  = 1 if setup['SWI_sites'] == 0 else pfas['f_F_sw']
    k_aw_f = pfas['f_k_aw_ave']/hour
    s_aw_f = pfas['f_k_aw_std']/hour
    k_sw_f = pfas['f_k_sw_ave']/hour
    s_sw_f = pfas['f_k_sw_std']/hour
    
    
    ## read immobile zone data
    wim = 1 - wf
    n_vg_im = soil['im_n_VG']  # VG parameter
    m_vg_im = 1 - 1/n_vg_im    # VG parameter 
    alpha_VG_im = soil['im_alpha_VG']/100 # VG parameter 
    poro_im = soil['im_theta_s']   # [-] 
    rhob_im = soil['im_rhob']   # g/cm^3
    kd_im = pfas['im_kd']    # cm^3/g
    if kd_im == 0:
        kf_im, N_im = pfas['im_kf'], pfas['im_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_im = kf_im * np.power(c_rep,N_im-1) # L/kg -> cm^3/g  
    kd_im *= setup['SWI_type']
    mu_im  = pfas['f_mu']/hour
    Faw_im = 1 if setup['AWI_sites'] == 0 else pfas['im_F_aw']  
    Fsw_im  = 1 if setup['SWI_sites'] == 0 else pfas['im_F_sw']
    k_aw_im = pfas['im_k_aw_ave']/hour # assume the same in each immobile subdomain
    s_aw_im = pfas['im_k_aw_std']/hour # assume the same in each immobile subdomain
    k_sw_im = pfas['im_k_sw_ave']/hour # assume the same in each immobile subdomain
    s_sw_im = pfas['im_k_sw_std']/hour # assume the same in each immobile subdomain
    
    # solve Se and Pc
    x= sp.Symbol('x') # Define symbolic variables
    q = x**0.5 * (1 - (1 - x**(1/m_vg_f))**m_vg_f)**2 * soil['f_ksat']
    se_f = float(sp.nsolve(I0 - soil['f_w']*q, 0.5))
    Pc =  (se_f**(-1/m_vg_f) - 1.0)**(1/n_vg_f)/alpha_VG_f
    theta_f = soil['f_theta_r'] + (soil['f_theta_s'] - soil['f_theta_r'])*se_f
    Aaw_f = Aaw_func_thermo(sigma0, poro_f, soil['f_alpha_VG'], n_vg_f, theta_f, soil['f_theta_r'], soil['f_theta_s'], soil['f_sf'])
    v_f =  I0/theta_f/day # cm/day -> cm/s
    # Note: Here  L/v_f is used for the nondimenionalization, instead of v_ave as reported in the publication.
    Pe_f = v_f*L / (v_f * alphaL_f + tau_f * D0) 
    
    
    # fracture - redardation factor
    Rf = np.array([1+Faw_f*Aaw_f*Kaw/theta_f + Fsw_f*kd_f*rhob_f/theta_f])
    # fracture - solid-water interface
    if setup['SWI_sites'] != 2:
        k_sw_fs =  np.array([k_sw_f])
        f_sw_fs =  np.array([1.])
    else:
        num_domains = int(setup['Nsw'])
        mu_ln = np.log(k_sw_f**2 / np.sqrt(k_sw_f**2 + s_sw_f**2))
        sigma_ln = np.sqrt(np.log((k_sw_f**2 + s_sw_f**2) /k_sw_f**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_sw_fs = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_sw_fs = lognorm.pdf(k_sw_fs, sigma_ln, scale=np.exp(mu_ln))
        f_sw_fs /= np.sum(f_sw_fs)    
    # fracture - air-water interface
    if setup['AWI_sites'] != 2:
        k_aw_fs =  np.array([k_aw_f])
        f_aw_fs =  np.array([1.]) 
    else:
        num_domains = int(setup['Naw'])
        mu_ln = np.log(k_aw_f**2 / np.sqrt(k_aw_f**2 + s_aw_f**2))
        sigma_ln = np.sqrt(np.log((k_aw_f**2 + s_aw_f**2) /k_aw_f**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_aw_fs = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_aw_fs = lognorm.pdf(k_aw_fs, sigma_ln, scale=np.exp(mu_ln))
        f_aw_fs /= np.sum(f_aw_fs)
          
        
    # immobile zones
    Nim  = int(setup['ImZones'])
    mu_ln = np.log(pfas['fim_kappa_ave']**2 / np.sqrt(pfas['fim_kappa_ave']**2 + pfas['fim_kappa_std']**2))
    sigma_ln = np.sqrt(np.log((pfas['fim_kappa_ave']**2 + pfas['fim_kappa_std']**2) /pfas['fim_kappa_ave']**2))
    lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
    ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
    kapa_fim = np.linspace(np.exp(lb), np.exp(ub), Nim)
    w_ims = lognorm.pdf(kapa_fim, sigma_ln, scale=np.exp(mu_ln))
    w_ims /= w_ims.sum()
    se_im = ( 1.0+(alpha_VG_im * Pc)**n_vg_im ) ** (-m_vg_im) 
    theta_im = (soil['im_theta_r'] + (soil['im_theta_s'] - soil['im_theta_r'])*se_im)
    theta_ims = np.ones(Nim) * theta_im # a vector if theta_im differs among immobile zones
    Aaw_im = Aaw_func_thermo(sigma0, poro_im, soil['im_alpha_VG'], n_vg_im, theta_im, soil['im_theta_r'], soil['im_theta_s'], soil['im_sf'])
    Faw_ims = np.ones(Nim) * Faw_im
    Fsw_ims = np.ones(Nim) * Fsw_im
    Rim = np.ones(Nim) + np.ones(Nim) * (Faw_ims*Aaw_im*Kaw + Fsw_ims*kd_im*rhob_im) / theta_ims
    mu_ims = np.ones(Nim) * mu_im
    k_aw_ims = np.ones(Nim) * k_aw_im # need revisions if varying adsorption rate constants are needed 
    k_sw_ims = np.ones(Nim) * k_sw_im # need revisions if varying adsorption rate constants are needed 
    

    # dimensionless parameters
    mu_f *= L/v_f
    mu_f *= setup['Transform']
    R_sw_f = rhob_f / theta_f
    A_sw_f = L * k_sw_fs / v_f * f_sw_fs * (1.0 - Fsw_f) * kd_f
    B_sw_f = L * k_sw_fs / v_f
    R_aw_f = Aaw_f / theta_f
    A_aw_f = L * k_aw_fs / v_f * f_aw_fs * (1.0 - Faw_f) * Kaw
    B_aw_f = L * k_aw_fs / v_f 
    Kp_fim =  wim * w_ims * kapa_fim * L / (wf * theta_f * v_f)
    mu_ims *= L/v_f
    mu_ims *= setup['Transform']
    Kp_ims = kapa_fim * L / ( theta_ims * v_f)
    R_sw_ims = rhob_im / theta_ims
    A_sw_ims = L * k_sw_ims / v_f * (1.0 - Fsw_ims) * kd_im
    B_sw_ims = L * k_sw_ims / v_f
    R_aw_ims = Aaw_im / theta_ims
    A_aw_ims = L * k_aw_ims / v_f * (1.0 - Faw_ims) * Kaw
    B_aw_ims = L * k_aw_ims / v_f
    
    
    # run simulations
    X, dT, T = 1, setup['dT'], setup['T']
    
    ######################
    # Analytical solution#
    ######################
    if setup['run_num'] in [0, 2]:
        print("Running analytical model.....")
        
        param = [Rf,  Pe_f, mu_f, Kp_fim, 
                 R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
                 Rim, Kp_ims, mu_ims, 
                 R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims]
        
        a, NSUM = setup['IL_a']/T, setup['IL_Nsum'] # may need to tune 'a' for minimizing the error
        
        time, conc = [], [] 
        time.append(0)
        conc.append(0/cin)
        dt, dx = T/setup['dt_num'], X/setup['dx_num']
        nt = 0
        for t in np.arange(0.0, T, dt): 
            if nt%int(T/dt/10) == 0:
                print(f"current time step is {t:.2f} [-], total time is {T:.2f} [-]!")
            nt += 1
            c = QE_method(t, X, dT, T, cin, param, setup, a/T, int(NSUM/2), 0)
            #c = DurbinMethod(t, X, dT, T, cin, param, setup, a, NSUM)
            #c = DurbinMethod_Wynn(t, X, dT, T, cin, param, setup, a, NSUM)
            time.append(t * (L/v_f)/year)
            conc.append(c[0]/cin)
        df = pd.DataFrame({"time": time, "conc": conc})
        df.to_csv("output/btc-anal.csv", index = False)
        print("Analytical solution ended!\n")
    
    #####################
    # Numerical solution#
    #####################
    if setup['run_num'] in [1, 2]:
        
        print("Running numerical simulation...")
        
        dt, dx = T/setup['dt_num'], X/setup['dx_num']
        nx = int(X/dx)
        N = nx
        # initial conditions
        c0 = np.zeros(N) # initial concentration in the advective domain
        c = np.zeros(N)
        c0_f_aw = np.zeros((N, f_aw_fs.shape[0])) # initial concentration at the air-water interfaces
        c0_f_sw = np.zeros((N,f_sw_fs.shape[0])) # initial concentration at the solid-water interfaces
        c0_ims = np.zeros((N, Nim))
        c0_im_aw = np.zeros((N, Nim)) # initial concentration at the air-water interfaces
        c0_im_sw = np.zeros((N, Nim)) # initial concentration at the solid-water interfaces
        
        # build Jacobian
        Jac = np.zeros(N**2)
        idx = np.arange(N,N*(N-1),N) + np.arange(1,N-1,1)
        
        # Fracture domain
        Jac[idx] = 2/Pe_f/dx/dx-1/dx + Rf/dt + mu_f # at x
        if setup['SWI_sites'] !=  0: 
            Jac[idx] += (R_sw_f*A_sw_f/(1+dt*B_sw_f)).sum()
        if setup['AWI_sites'] !=  0: 
            Jac[idx] += (R_aw_f*A_aw_f/(1+dt*B_aw_f)).sum()    
        Jac[idx - 1] = (-1/Pe_f/dx/dx) # X - dx
        Jac[idx + 1] = (-1/Pe_f/dx/dx+1/dx) # X + dx
            
        # Immobile matrix domain
        if wim > 0 and wim < 1: 
            Sims = Rim/dt + Kp_ims + mu_ims 
            if setup['SWI_sites'] !=  0: 
                Sims += R_sw_ims * A_sw_ims / (1+dt*B_sw_ims) 
            if setup['AWI_sites'] !=  0: 
                Sims += R_aw_ims * A_aw_ims / (1+dt*B_aw_ims)   
        if wim > 0 and wim < 1:
            Jac[idx] += (Kp_fim * (Sims - Kp_ims) / Sims).sum()
            
        # Boundaries
        Jac[0], Jac[1] = (1+1/Pe_f/dx), (-1/Pe_f/dx) # inlet
        Jac[(N+1)*(N-1)], Jac[(N+1)*(N-1)-1] = -1, 1 # outlet
        Jacm = Jac.reshape(N, N)
        
        # time marching
        time, conc = [], [] 
        time.append(0)
        conc.append(c[-1]/cin)
        nt = 0
        for t in np.arange(0.0, T, dt): #obs[:,0]:
            if nt%int(T/dt/10) == 0:
                print(f"current time step is {t:.2f} [-], total time is {T:.2f} [-]!")
            nt += 1
            # right hand side
            b = Rf/dt * c0
            if setup['SWI_sites'] !=  0: 
                b += ( c0_f_sw*((R_sw_f*B_sw_f)/(1+dt*B_sw_f)) ).sum(axis=1) 
            if setup['AWI_sites'] !=  0:
                b+= ( c0_f_aw*((R_aw_f*B_aw_f)/(1+dt*B_aw_f)) ).sum(axis=1)
                
            if wim > 0 and wim < 1:
                b += ( (c0_ims*(Rim/dt)) * (Kp_fim/Sims) ).sum(axis=1)
                if setup['SWI_sites'] !=  0: 
                    b += ( (c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims))) * (Kp_fim/Sims) ).sum(axis=1) 
                if setup['AWI_sites'] !=  0: 
                    b += ( (c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims))) * (Kp_fim/Sims) ).sum(axis=1)
                
            # boundary condition
            b[0] = cin if t <= dT else 0 
            b[N-1] = 0
            # solve
            c = np.linalg.solve(Jacm, b)
            # update previous time step
            c0 = c
            if setup['SWI_sites'] !=  0: 
                c0_f_sw = c0_f_sw/(1+dt*B_sw_f) + c[:, np.newaxis]*((dt*A_sw_f)/(1+dt*B_sw_f))
            if setup['AWI_sites'] !=  0: 
                c0_f_aw = c0_f_aw/(1+dt*B_aw_f) + c[:, np.newaxis]*((dt*A_aw_f)/(1+dt*B_aw_f))
                
            if wim > 0 and wim < 1: # Add immobile zone
                c0_ims = ( c[:, np.newaxis]*Kp_ims + c0_ims*(Rim/dt) ) / Sims  
                if setup['SWI_sites'] !=  0: 
                    c0_ims += ( c0_im_sw*((R_sw_ims*B_sw_ims)/(1+dt*B_sw_ims)) ) / Sims
                    c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims)) # update kenitic SWI conc
                if setup['AWI_sites'] !=  0:    
                    c0_ims += ( c0_im_aw*((R_aw_ims*B_aw_ims)/(1+dt*B_aw_ims)) ) / Sims
                    c0_im_sw = c0_im_sw*(1/(1+dt*B_sw_ims)) + c0_ims*((dt*A_sw_ims)/(1+dt*B_sw_ims)) # update kenitic SWI conc
                    c0_im_aw = c0_im_aw*(1/(1+dt*B_aw_ims)) + c0_ims*((dt*A_aw_ims)/(1+dt*B_aw_ims)) # update kenitic AWI conc
            # store results
            time.append(t * (L/v_f)/year)
            conc.append(c[-1]/cin)
        df = pd.DataFrame({"time": time, "conc": conc})
        df.to_csv("output/btc-num.csv", index = False)
        print("Numerical simulation ended!\n")

def semi_analytical_solution(setup, soil, pfas):
    print("The model is being reorganized and is expected to arrive soon....")
