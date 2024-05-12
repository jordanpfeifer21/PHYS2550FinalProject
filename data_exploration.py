import numpy as np
import pandas as pd

def calculate_invariant_mass(df):
    momentum_col = df.columns[:700]
    eta_col = df.columns[700:1400]
    phi_col = df.columns[-700:]
    energy_total = .0
    momentum_total = .0

    for ind in np.arange(0, len(momentum_col), step=1):
        p_x = df[momentum_col[ind]] * np.cos(df[phi_col[ind]])
        p_y = df[momentum_col[ind]] * np.sin(df[phi_col[ind]])
        p_z = df[momentum_col[ind]] * np.sinh(df[eta_col[ind]])
        energy = np.sqrt(df[momentum_col[ind]] ** 2 + p_z ** 2)
        energy_total += energy
        momentum_total += (p_x + p_y + p_z)
    
    print(f'here: {energy_total[:10]}, and {momentum_total[:10]}')
    
    return np.sqrt(energy_total ** 2 - momentum_total ** 2)


def calculate_sum_transverse_momentum(df):
    momentum_col = df.columns[:700]
    momentum_total = .0

    for column in momentum_col:
        momentum_total += df[column]

    return momentum_total