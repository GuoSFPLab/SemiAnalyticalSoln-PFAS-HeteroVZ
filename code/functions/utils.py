import pandas as pd
from .solutions import analytical_solution, semi_analytical_solution

def run_simulation(setupfile, soilfile, pfasfile):
    setup = read_data(setupfile)
    soil  = read_data(soilfile)
    pfas  = read_data(pfasfile)
    
    checks = {
                0: {
                    "msg": "Running single porosity model...\n",
                    "end": "Finish single porosity simulation!!!!!!",
                    "valid": lambda s: s['f_w'] == 1.0,
                    "error": "Invalid f_w, f_w != 1",
                    "solution": analytical_solution
                },
                1: {
                    "msg": "Running dual porosity model...\n",
                    "end": "Finish dual porosity simulation!!!!!!",
                    "valid": lambda s: s['f_w'] + s['im_w'] == 1.0,
                    "error": "Invalid f_w or im_w, f_w + im_w != 1",
                    "solution": analytical_solution
                },
                2: {
                    "msg": "Running dual permeability model...\n",
                    "end": "Finish dual permeability simulation!!!!!!",
                    "valid": lambda s: s['f_w'] + s['m_w'] == 1.0,
                    "error": "Invalid f_w or m_w, f_w + m_w != 1",
                    "solution": semi_analytical_solution
                },
                3: {
                    "msg": "Running triple porosity model...\n",
                    "end": "Finish triple porosity simulation!!!!!!",
                    "valid": lambda s: s['f_w'] + s['m_w'] + s['im_w'] == 1.0,
                    "error": "Invalid f_w or m_w or im_w, f_w + m_w + im_w != 1",
                    "solution": semi_analytical_solution
                }
            }   
    
    if setup['perm_domain'] in checks:
        check = checks[setup['perm_domain']]
        print(check["msg"])
        if check["valid"](soil):
            check["solution"](setup, soil, pfas)
        else:
            print(check["error"])
    else:
        print(f"Invalid 'perm_domain' value: {setup['perm_domain']}, expected 0, 1, 2, or 3.")
    print(check["end"])
    

def read_data(filename, as_dict=True):
    """
    Reads a CSV like your parameter files.
    - Second column becomes the header.
    - First column becomes the value row.
    - If as_dict=True, return a dict (scalar access) instead of DataFrame.
    """
    df = pd.read_csv(filename, header=None)
    df[1] = df[1].str.strip()  # clean header
    
    header = df.iloc[:, 1].tolist()
    values = df.iloc[:, 0].tolist()
    
    df_new = pd.DataFrame([values], columns=header)
    
    if as_dict:
        return df_new.iloc[0].to_dict()  # return single row as dict
    return df_new
