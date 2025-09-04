"""
This script runs a PFAS transport simulation in homogeneous and heterogeneous vadose zones 
based on the provided model setup, soil properties, and PFAS parameters. It then reads the 
breakthrough curves from the output CSV file and plots the normalized concentration versus time.
Both analytical and numerical solutions are available for the following four model complexities: 
      (1) Single-porosity model,
      (2) Dual-porosity model [Note: L/v_f (instead of L/v_ave) is used for nondimensionalization.],
      (3) Dual-permeability model,
      (4) Triple-porosity model.
      
Two example cases are provided: (1) A dual-porosity case, & (2) A triple-porosity case. Users can customize 
the input files in the "input" folder for their simulations. However, they may adjust "IL_a" & "IL_Nsum" 
in the "model_setup*.txt" to reduce numerical integration errors of the semi-analytical solutions, or 
increase "dt_num" & "dx_num" in the same file to reduce discretization errors of the numerical solutions.

Author: Sidian Chen (Phd @University of Arizona, now postdoc @Stanford University)
Email: sidianc@stanford.edu
Date: 09/02/2025

Reference: Chen, S., & Guo, B.(2025) Semi-analytical solutions for nonequilibrium transport and 
           transformation of PFAS and other solutes in heterogeneous vadose zones with structured 
           porous media. Advances in Water Resources.
"""

from functions.utils import run_simulation, read_data
import matplotlib.pyplot as plt
import pandas as pd
import glob


# input & output folders
input_folder, output_folder = 'input', 'output'

# run the two example cases
for casename, msg in [("DualPoro",   "Case 1: Dual-porosity model"),
                      ("TriplePoro", "Case 2: Triple-porosity model")]:
    # print casename
    print("*************************************************************")
    print("*************************************************************")
    print("*************************************************************")
    print(f'{msg} starts...')
    print("*************************************************************\n")

    # names of input files for model setup, soil properties, and PFAS properties
    filenames = [glob.glob(f"{input_folder}/*model_setup*{casename}*.txt")[0], 
                 glob.glob(f"{input_folder}/*soil_properties*{casename}*.csv")[0], 
                 glob.glob(f"{input_folder}/*pfas_parameters*.csv")[0]]
    
    # run simulations (Not required if results already exist)
    run_simulation(filenames)
    
    # read results from output file and plot the results 
    setup = read_data(filenames[0])
    plt.figure(figsize=(6,4))
    # analytical solution
    if setup['run_num'] in [0, 1, 2]:
        scalar = 4
        df = pd.read_csv(f"{output_folder}/btc-anal.csv")
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
        plt.plot(df['time'], df['conc'], lw = 2, 
                 label='Analytical solution' if setup['perm_domain'] <= 1 else 'Analytical solution (Fracture)')
        if setup['perm_domain'] >= 2:
            plt.plot(df['time'], df['concm'], lw = 2, label='Analytical solution (Matrix)')
            plt.plot(df['time'], df['conc_ave'], lw = 2, label='Analytical solution (Average)')
        
    # numerical solution
    if setup['run_num'] in [0, 1, 2]:
        df = pd.read_csv(f"{output_folder}/btc-num.csv")
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
        plt.plot(df['time'], df['conc'], lw = 3, linestyle='--', 
                 label='Numerical solution' if setup['perm_domain'] <= 1 else 'Numerical solution (Fracture)')
        if setup['perm_domain'] >= 2:
            plt.plot(df['time'], df['concm'], lw = 3, linestyle='--', label='Numerical solution (Matrix)')
            plt.plot(df['time'], df['conc_ave'], lw = 3, linestyle='--', label='Numerical solution (Average)')
    
    # set axis ticks and labels
    plt.xlabel("Time (year)")
    plt.ylabel("Normalized concentration (C/C$_0$)")
    plt.title(f"Breakthrough Curve ({msg})")
    plt.tick_params(axis="both", direction="in")
    plt.legend(frameon=False)
    plt.savefig(f"{output_folder}/breakthrough{casename}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    print("*************************************************************")
    print(f'{msg} has been compelted!')
    print("*************************************************************")
    print("*************************************************************")
    print("*************************************************************\n\n")
