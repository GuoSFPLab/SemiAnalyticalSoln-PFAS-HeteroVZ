"""
This script runs a PFAS transport simulation in homogeneous and heterogeneous vadose zones 
based on provided model setup, soil properties, and PFAS parameters. It then reads the 
breakthrough curves from the output CSV file and plots the normalized concentration versus time.
Both analytical and numerical solutions are available.

Author: Sidian Chen (Phd @ University of Arizona, now postdoc @ Stanford University)
Email: sidianc@stanford.edu
Date: 09/01/2025

Reference: Chen, S. and Guo, B.(2025) Semi-analytical solutions for nonequilibrium transport and 
           transformation of PFAS and other solutes in heterogeneous vadose zones with structured 
           porous media. Advances in Water Resources.
      
Note: In the dual-porosity model, L/v_f (instead of L/v_ave) is used for nondimensionalization.
"""

from functions.utils import run_simulation, read_data
import matplotlib.pyplot as plt
import pandas as pd

# run simulations (Not required if results alread exist)
# run_simulation("input/model_setup.txt", "input/soil_properties.csv", "input/pfas_parameters.csv")

# read results from output file and plot the results 
setup = read_data("input/model_setup.txt")
plt.figure(figsize=(6,4))

# analytical solution
if setup['run_num'] in [0, 2]:
    df = pd.read_csv("output/btc-anal.csv")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    plt.plot(df['time'], df['conc'], lw = 2, label='Analytical solution')
    
# numerical solution
if setup['run_num'] in [1, 2]:
    df = pd.read_csv("output/btc-num.csv")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")
    plt.plot(df['time'], df['conc'], lw = 3, linestyle='--', label='Numerical solution')

# set axis ticks and labels
plt.xlabel("Time (year)")
plt.ylabel("Normalized concentration (C/C$_0$)")
plt.title("Breakthrough Curve")
plt.tick_params(axis="both", direction="in")
plt.legend(frameon=False)
plt.savefig("output/breakthrough.png", dpi=300, bbox_inches="tight")
plt.show()