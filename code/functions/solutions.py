import numpy as np
from scipy.stats import lognorm
import pandas as pd
import sympy as sp
from .analytical_solvers import QE_method, QE_method_TP
from .numerical_solvers import num_SinglePoroDualPoro, num_DualPermTriPoro

##################################################################################################
##################################################################################################
##################################################################################################
##################################### Supplementary funtions #####################################
##################################################################################################
##################################################################################################
##################################################################################################
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


##################################################################################################
##################################################################################################
##################################################################################################
############################# Single-porosity & dual-porosity models #############################
##################################################################################################
##################################################################################################
##################################################################################################
def SinglePoro_DualPoro(setup, soil, pfas):
    
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
    Kaw = setup['f_AWI_type'] * 0.1*M0*sigma0*b/(Rg*Tp*(a+c_rep)) # [cm]    
    D0  = pfas['D0'] #cm^2/s
    
    ## read fracture data
    wf = soil['f_w']
    n_vg_f = soil['f_n_VG']  # VG parameter
    m_vg_f = 1 - 1/n_vg_f    # VG parameter 
    alpha_vg_f = soil['f_alpha_VG']/100 # VG parameter 
    alphaL_f = soil['f_alpha_L']
    poro_f = soil['f_theta_s']   # [-] 
    rhob_f = soil['f_rhob']   # g/cm^3
    tau_f =  soil['f_tau']
    kd_f = pfas['f_kd']    # cm^3/g
    if kd_f == 0:
        kf_f, N_f = pfas['f_kf'], pfas['f_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_f = kf_f * np.power(c_rep,N_f-1) # L/kg -> cm^3/g  
    kd_f *= setup['f_SWI_type']
    mu_f  = pfas['f_mu']/hour
    Faw_f = 1 if setup['f_AWI_sites'] == 0 else pfas['f_F_aw']
    k_aw_f = pfas['f_k_aw_ave']
    Fsw_f  = 1 if setup['f_SWI_sites'] == 0 else pfas['f_F_sw']
    k_aw_f = pfas['f_k_aw_ave']/hour
    s_aw_f = pfas['f_k_aw_std']/hour
    k_sw_f = pfas['f_k_sw_ave']/hour
    s_sw_f = pfas['f_k_sw_std']/hour
    
    
    ## read immobile zone data
    wim = 1 - wf
    n_vg_im = soil['im_n_VG']  # VG parameter
    m_vg_im = 1 - 1/n_vg_im    # VG parameter 
    alpha_vg_im = soil['im_alpha_VG']/100 # VG parameter 
    poro_im = soil['im_theta_s']   # [-] 
    rhob_im = soil['im_rhob']   # g/cm^3
    kd_im = pfas['im_kd']    # cm^3/g
    if kd_im == 0:
        kf_im, N_im = pfas['im_kf'], pfas['im_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_im = kf_im * np.power(c_rep,N_im-1) # L/kg -> cm^3/g  
    kd_im *= setup['f_SWI_type']
    mu_im  = pfas['f_mu']/hour
    Faw_im = 1 if setup['f_AWI_sites'] == 0 else pfas['im_F_aw']  
    Fsw_im  = 1 if setup['f_SWI_sites'] == 0 else pfas['im_F_sw']
    k_aw_im = pfas['im_k_aw_ave']/hour # assume the same in each immobile subdomain
    k_sw_im = pfas['im_k_sw_ave']/hour # assume the same in each immobile subdomain
    
    # solve Se and Pc
    x= sp.Symbol('x') # Define symbolic variables
    q = x**0.5 * (1 - (1 - x**(1/m_vg_f))**m_vg_f)**2 * soil['f_ksat']
    se_f = float(sp.nsolve(I0 - soil['f_w']*q, 0.5))
    Pc =  (se_f**(-1/m_vg_f) - 1.0)**(1/n_vg_f)/alpha_vg_f
    theta_f = soil['f_theta_r'] + (soil['f_theta_s'] - soil['f_theta_r'])*se_f
    Aaw_f = Aaw_func_thermo(sigma0, poro_f, soil['f_alpha_VG'], n_vg_f, theta_f, soil['f_theta_r'], soil['f_theta_s'], soil['f_sf'])
    v_f =  I0/theta_f/day # cm/day -> cm/s
    # Note: Here  L/v_f is used for the nondimenionalization, instead of v_ave as reported in the publication.
    Pe_f = v_f*L / (v_f * alphaL_f + tau_f * D0) 
    
    
    # fracture - redardation factor
    Rf = np.array([1+Faw_f*Aaw_f*Kaw/theta_f + Fsw_f*kd_f*rhob_f/theta_f])
    # fracture - solid-water interface
    if setup['f_SWI_sites'] != 2:
        k_sw_fs =  np.array([k_sw_f])
        f_sw_fs =  np.array([1.])
    else:
        num_domains = int(setup['f_Nsw'])
        mu_ln = np.log(k_sw_f**2 / np.sqrt(k_sw_f**2 + s_sw_f**2))
        sigma_ln = np.sqrt(np.log((k_sw_f**2 + s_sw_f**2) /k_sw_f**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_sw_fs = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_sw_fs = lognorm.pdf(k_sw_fs, sigma_ln, scale=np.exp(mu_ln))
        f_sw_fs /= np.sum(f_sw_fs)    
    # fracture - air-water interface
    if setup['f_AWI_sites'] != 2:
        k_aw_fs =  np.array([k_aw_f])
        f_aw_fs =  np.array([1.]) 
    else:
        num_domains = int(setup['f_Naw'])
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
    se_im = ( 1.0+(alpha_vg_im * Pc)**n_vg_im ) ** (-m_vg_im) 
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
    dt = T/setup['dt_num']
    param = [Rf,  Pe_f, mu_f, Kp_fim, 
             R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
             Rim, Kp_ims, mu_ims, 
             R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims]
    ######################
    # Analytical solution#
    ######################
    if setup['run_num'] in [0, 2]:
        print("Running analytical model.....")
        
        a, NSUM = setup['IL_a']/T, setup['IL_Nsum'] # may need to tune 'a' for minimizing the error
        
        time, conc = [], [] 
        time.append(0)
        conc.append(0/cin)
        nt = 0
        # time series
        for t in np.arange(0.0, T+dt, dt): 
            if nt%int(T/dt/10) == 0 or t > T-dt:
                print(f"current time step is {t:.2f} [PV], total time is {T:.2f} [PV]!")
            nt += 1
            c = QE_method(t, X, dT, T, cin, param, setup, a/T, int(NSUM), 0)
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
        thetas =  [wf, theta_f, wim, theta_im, (L/v_f)/year]
        time, conc = num_SinglePoroDualPoro(X, dT, T, dt, cin, param, thetas, setup)
        df = pd.DataFrame({"time": time, "conc": conc})
        df.to_csv("output/btc-num.csv", index = False)
        print("Numerical simulation ended!\n")

##################################################################################################
##################################################################################################
##################################################################################################
########################## Dual-permeability & triple-porosity models ############################
##################################################################################################
##################################################################################################
##################################################################################################
def DualPerm_TriPoro(setup, soil, pfas):

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
    Kaw = setup['f_AWI_type'] * 0.1*M0*sigma0*b/(Rg*Tp*(a+c_rep)) # [cm]    
    D0  = pfas['D0'] #cm^2/s
    
    ## read fracture data
    wf = soil['f_w']
    n_vg_f = soil['f_n_VG']  # VG parameter
    m_vg_f = 1 - 1/n_vg_f    # VG parameter 
    alpha_vg_f = soil['f_alpha_VG']/100 # VG parameter 
    alphaL_f = soil['f_alpha_L']
    poro_f = soil['f_theta_s']   # [-] 
    rhob_f = soil['f_rhob']   # g/cm^3
    tau_f =  soil['f_tau']
    kd_f = pfas['f_kd']    # cm^3/g
    if kd_f == 0:
        kf_f, N_f = pfas['f_kf'], pfas['f_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_f = kf_f * np.power(c_rep,N_f-1) # L/kg -> cm^3/g  
    kd_f *= setup['f_SWI_type']
    mu_f  = pfas['f_mu']/hour
    Faw_f = 1 if setup['f_AWI_sites'] == 0 else pfas['f_F_aw']
    k_aw_f = pfas['f_k_aw_ave']
    Fsw_f  = 1 if setup['f_SWI_sites'] == 0 else pfas['f_F_sw']
    k_aw_f = pfas['f_k_aw_ave']/hour
    s_aw_f = pfas['f_k_aw_std']/hour
    k_sw_f = pfas['f_k_sw_ave']/hour
    s_sw_f = pfas['f_k_sw_std']/hour
    Kp_fm = pfas['fm_kappa']/hour
    Kp_mf = pfas['fm_kappa']/hour
    
    ## read mobile matrix data
    wm = soil['m_w']
    n_vg_m = soil['m_n_VG']  # VG parameter
    m_vg_m = 1 - 1/n_vg_m    # VG parameter 
    alpha_vg_m = soil['m_alpha_VG']/100 # VG parameter 
    alphaL_m = soil['m_alpha_L']
    poro_m = soil['m_theta_s']   # [-] 
    rhob_m = soil['m_rhob']   # g/cm^3
    tau_m =  soil['m_tau']
    kd_m = pfas['m_kd']    # cm^3/g
    if kd_m == 0:
        kf_m, N_m = pfas['m_kf'], pfas['m_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_m = kf_m * np.power(c_rep,N_m-1) # L/kg -> cm^3/g  
    kd_m *= setup['m_SWI_type']
    mu_m  = pfas['m_mu']/hour
    Faw_m = 1 if setup['m_AWI_sites'] == 0 else pfas['m_F_aw']
    k_aw_m = pfas['m_k_aw_ave']
    Fsw_m  = 1 if setup['m_SWI_sites'] == 0 else pfas['m_F_sw']
    k_aw_m = pfas['m_k_aw_ave']/hour
    s_aw_m = pfas['m_k_aw_std']/hour
    k_sw_m = pfas['m_k_sw_ave']/hour
    s_sw_m = pfas['m_k_sw_std']/hour
    
    
    ## read immobile zone data
    wim = 1 - wf - wm
    n_vg_im = soil['im_n_VG']  # VG parameter
    m_vg_im = 1 - 1/n_vg_im    # VG parameter 
    alpha_vg_im = soil['im_alpha_VG']/100 # VG parameter 
    poro_im = soil['im_theta_s']   # [-] 
    rhob_im = soil['im_rhob']   # g/cm^3
    kd_im = pfas['im_kd']    # cm^3/g
    if kd_im == 0:
        kf_im, N_im = pfas['im_kf'], pfas['im_N']    # (mg/kg)*(mg/L)^(-N), [-]
        kd_im = kf_im * np.power(c_rep,N_im-1) # L/kg -> cm^3/g  
    kd_im *= setup['f_SWI_type']
    mu_im  = pfas['f_mu']/hour
    Faw_im = 1 if setup['f_AWI_sites'] == 0 else pfas['im_F_aw']  
    Fsw_im  = 1 if setup['f_SWI_sites'] == 0 else pfas['im_F_sw']
    k_aw_im = pfas['im_k_aw_ave']/hour # assume the same in each immobile subdomain
    k_sw_im = pfas['im_k_sw_ave']/hour # assume the same in each immobile subdomain
    
    # solve Se and Pc
    thetar_f, thetas_f = soil['f_theta_r'], soil['f_theta_s']
    thetar_m, thetas_m = soil['m_theta_r'], soil['m_theta_s']
    def solvePc():
        import sympy as sp
        # Define symbolic variables
        x = sp.Symbol('x')
        theta_f = thetar_f + (thetas_f-thetar_f)/( 1.0+(alpha_vg_f * x)**n_vg_f ) ** m_vg_f
        theta_m = thetar_m + (thetas_m-thetar_m)/( 1.0+(alpha_vg_m * x)**n_vg_m ) ** m_vg_m
        se_f = (theta_f - thetar_f) / (thetas_f - thetar_f)
        se_m = (theta_m - thetar_m) / (thetas_m - thetar_m)
        q_f = se_f**0.5 * (1 - (1 - se_f**(1/m_vg_f))**m_vg_f)**2 * soil['f_ksat']
        q_m = se_m**0.5 * (1 - (1 - se_m**(1/m_vg_m))**m_vg_m)**2 * soil['m_ksat']
        initial_guess = -soil['GuessPc']                         # initial guess of absolute capillary pressure [cm]
        equation = wf * q_f + wm * q_m - I0
        Pc = sp.nsolve(equation, initial_guess)
        return float(Pc)
    Pc = - solvePc()
    
    
    # fracture - hydraulic
    se_f = (1.0 + (alpha_vg_f * abs(Pc))**n_vg_f) ** (-m_vg_f)
    theta_f = thetar_f + (thetas_f - thetar_f)*se_f
    Aaw_f = Aaw_func_thermo(sigma0, poro_f, alpha_vg_f, n_vg_f, theta_f, thetar_f, thetas_f, soil['f_sf'])
    v_f =  se_f**0.5 * (1 - (1 - se_f**(1/m_vg_f))**m_vg_f)**2 * soil['f_ksat']/theta_f/day # cm/day -> cm/s
    # Note: Here  L/v_f is used for the nondimenionalization, instead of v_ave as reported in the publication.
    Pe_f = v_f*L / (v_f * alphaL_f + tau_f * D0) 
    
    # fracture - redardation factor
    Rf = np.array([1+Faw_f*Aaw_f*Kaw/theta_f + Fsw_f*kd_f*rhob_f/theta_f])
    
    # fracture - solid-water interface
    if setup['f_SWI_sites'] != 2:
        k_sw_fs =  np.array([k_sw_f])
        f_sw_fs =  np.array([1.])
    else:
        num_domains = int(setup['f_Nsw'])
        mu_ln = np.log(k_sw_f**2 / np.sqrt(k_sw_f**2 + s_sw_f**2))
        sigma_ln = np.sqrt(np.log((k_sw_f**2 + s_sw_f**2) /k_sw_f**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_sw_fs = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_sw_fs = lognorm.pdf(k_sw_fs, sigma_ln, scale=np.exp(mu_ln))
        f_sw_fs /= np.sum(f_sw_fs)    
    # fracture - air-water interface
    if setup['f_AWI_sites'] != 2:
        k_aw_fs =  np.array([k_aw_f])
        f_aw_fs =  np.array([1.]) 
    else:
        num_domains = int(setup['f_Naw'])
        mu_ln = np.log(k_aw_f**2 / np.sqrt(k_aw_f**2 + s_aw_f**2))
        sigma_ln = np.sqrt(np.log((k_aw_f**2 + s_aw_f**2) /k_aw_f**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_aw_fs = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_aw_fs = lognorm.pdf(k_aw_fs, sigma_ln, scale=np.exp(mu_ln))
        f_aw_fs /= np.sum(f_aw_fs)
        
        
    # mobile matrix - hydraulic
    se_m = (1.0 + (alpha_vg_m * abs(Pc))**n_vg_m) ** (-m_vg_m)
    theta_m = thetar_m + (thetas_m - thetar_m)*se_m
    Aaw_m = Aaw_func_thermo(sigma0, poro_m, alpha_vg_m, n_vg_m, theta_m, thetar_m, thetas_m, soil['m_sf'])
    v_m =  se_m**0.5 * (1 - (1 - se_m**(1/m_vg_m))**m_vg_m)**2 * soil['m_ksat']/theta_m/day # cm/day -> cm/s
    # Note: Here  L/v_f is used for the nondimenionalization, instead of v_ave as reported in the publication.
    Pe_m = v_m*L / (v_m * alphaL_m + tau_m * D0) 
    
    # mobile matrix - redardation factor
    Rm = np.array([1+Faw_m*Aaw_m*Kaw/theta_m + Fsw_m*kd_m*rhob_m/theta_m])
    # mobile matrix - solid-water interface
    if setup['m_SWI_sites'] != 2:
        k_sw_ms =  np.array([k_sw_m])
        f_sw_ms =  np.array([1.])
    else:
        num_domains = int(setup['m_Nsw'])
        mu_ln = np.log(k_sw_m**2 / np.sqrt(k_sw_m**2 + s_sw_m**2))
        sigma_ln = np.sqrt(np.log((k_sw_m**2 + s_sw_m**2) /k_sw_m**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_sw_ms = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_sw_ms = lognorm.pdf(k_sw_ms, sigma_ln, scale=np.exp(mu_ln))
        f_sw_ms /= np.sum(f_sw_ms)    
    # mobile matrix - air-water interface
    if setup['m_AWI_sites'] != 2:
        k_aw_ms =  np.array([k_aw_m])
        f_aw_ms =  np.array([1.]) 
    else:
        num_domains = int(setup['m_Naw'])
        mu_ln = np.log(k_aw_f**2 / np.sqrt(k_aw_m**2 + s_aw_m**2))
        sigma_ln = np.sqrt(np.log((k_aw_m**2 + s_aw_m**2) /k_aw_m**2))
        lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
        ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
        k_aw_ms = np.linspace(np.exp(lb), np.exp(ub), num_domains)
        f_aw_ms = lognorm.pdf(k_aw_ms, sigma_ln, scale=np.exp(mu_ln))
        f_aw_ms /= np.sum(f_aw_ms)
          
        
    # immobile zones
    Nim  = int(setup['ImZones'])
    mu_ln = np.log(pfas['mim_kappa_ave']**2 / np.sqrt(pfas['mim_kappa_ave']**2 + pfas['mim_kappa_std']**2))
    sigma_ln = np.sqrt(np.log((pfas['mim_kappa_ave']**2 + pfas['mim_kappa_std']**2) /pfas['mim_kappa_ave']**2))
    lb = find_x_at_given_f(mu_ln, sigma_ln, 0)
    ub = find_x_at_given_f(mu_ln, sigma_ln, 1)
    kapa_mim = np.linspace(np.exp(lb), np.exp(ub), Nim)
    w_ims = lognorm.pdf(kapa_mim, sigma_ln, scale=np.exp(mu_ln))
    w_ims /= w_ims.sum()
    se_im = ( 1.0+(alpha_vg_im * abs(Pc))**n_vg_im ) ** (-m_vg_im) 
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
    theta = wf * theta_f + wm* theta_m + (w_ims*theta_ims).sum()
    v_ave = (wf*theta_f*v_f + wf*theta_f*v_f)/theta
    
    # fracture
    Rf *= v_ave/v_f
    mu_f *= L/v_f
    mu_f *= setup['Transform']
    R_sw_f = (rhob_f * v_ave) / (theta_f * v_f)
    A_sw_f = L * k_sw_fs / v_ave * f_sw_fs * (1.0 - Fsw_f) * kd_f
    B_sw_f = L * k_sw_fs / v_ave
    R_aw_f =(Aaw_f*v_ave) /( theta_f * v_f)
    A_aw_f = L * k_aw_fs / v_ave * f_aw_fs * (1.0 - Faw_f) * Kaw
    B_aw_f = L * k_aw_fs / v_ave 
    Kp_fm *= (L * wm) / (theta_f*v_f*wf)
    
    # mobile matrix
    Rm *= v_ave/v_m
    mu_m *= L/v_m
    mu_m *= setup['Transform']
    R_sw_m = (rhob_m * v_ave) / (theta_m * v_m)
    A_sw_m = L * k_sw_ms / v_ave * f_sw_ms * (1.0 - Fsw_m) * kd_m
    B_sw_m = L * k_sw_ms / v_ave
    R_aw_m =(Aaw_m*v_ave) /( theta_m * v_m)
    A_aw_m = L * k_aw_ms / v_ave * f_aw_ms * (1.0 - Faw_m) * Kaw
    B_aw_m = L * k_aw_ms / v_ave 
    Kp_mf *= L / (theta_m*v_m)
    Kp_mim =  wim * w_ims * kapa_mim * L / (wm * theta_m * v_m)
    

    # immobile mobile matrix
    mu_ims *= L/v_ave
    mu_ims *= setup['Transform']
    Kp_ims = kapa_mim * L / ( theta_ims * v_ave)
    R_sw_ims = rhob_im / theta_ims
    A_sw_ims = L * k_sw_ims / v_ave * (1.0 - Fsw_ims) * kd_im
    B_sw_ims = L * k_sw_ims / v_ave
    R_aw_ims = Aaw_im / theta_ims
    A_aw_ims = L * k_aw_ims / v_ave * (1.0 - Faw_ims) * Kaw
    B_aw_ims = L * k_aw_ims / v_ave
    
    
    # run simulations
    X, dT, T = 1, setup['dT'], setup['T']
    dt = T/setup['dt_num']
    
    param = [Rf,  Pe_f, mu_f, Kp_fm, 
              R_sw_f, A_sw_f, B_sw_f, R_aw_f,A_aw_f, B_aw_f, 
              Rm,  Pe_m, mu_m, Kp_mf, 
              R_sw_m, A_sw_m, B_sw_m, R_aw_m,A_aw_m, B_aw_m, Kp_mim,
              Rim, Kp_ims, mu_ims, 
              R_sw_ims, A_sw_ims, B_sw_ims, R_aw_ims, A_aw_ims, B_aw_ims]
    
    
    ######################
    # Analytical solution#
    ######################
    if setup['run_num'] in [0, 2]:
        print("Running analytical model.....")
        a, NSUM = setup['IL_a']/T, setup['IL_Nsum'] # may need to tune 'a' for minimizing the error
        dt = T/setup['dt_num']
        time, concf, concm, conc_ave = [], [], [], []
        nt = 0
        # time series
        for t in np.arange(0.0, T+dt, dt): 
            if nt%int(T/dt/10) == 0 or t > T-dt:
                print(f"current time step is {t:.2f} [PV], total time is {T:.2f} [PV]!")
            nt += 1
            c = QE_method_TP(t, X, dT, T, cin, param, setup, a/T, int(NSUM), 0) 
            time.append(t * (L/v_ave)/year)
            concf.append(c[0]/cin)
            concm.append(c[1]/cin)
            conc_ave.append((wf*theta_f*c[0] + wm*theta_m*c[1])/(wf*theta_f+wm*theta_m)/cin)  
        df = pd.DataFrame({"time": time, "conc": concf, "concm": concm, "conc_ave": conc_ave})
        df.to_csv("output/btc-anal.csv", index = False)
        print("Analytical solution ended!\n")
    
    #####################
    # Numerical solution#
    #####################
    if setup['run_num'] in [1, 2]:    
        print("Running numerical simulation...")
        thetas =  [wf, theta_f, wm, theta_m, wim, theta_im, (L/v_ave)/year]
        time, concf, concm, conc_ave = num_DualPermTriPoro(X, dT, T, dt, cin, param, thetas, setup)
        df = pd.DataFrame({"time": time, "conc": concf, "concm": concm, "conc_ave": conc_ave})
        df.to_csv("output/btc-num.csv", index = False)
        print("Numerical simulation ended!\n")