import numpy as np
import scipy as sp
from numba import njit, jit, prange




@njit(parallel=True)
def trial_wave_function_1(x, a, B):

    if np.abs(x) <= a:
        return B * (np.square(a) - np.square(x))
    
    else:
        return 0
    
    # return B * (np.square(a) - np.square(x))



@njit(parallel=True)
def trial_wave_function_2(x, beta, A):

    return A * np.exp(-1 * beta * np.square(x))



@njit(parallel=True)
def probability_density_1(x, a, B, trial_wave_function_1):
    
    return np.square(trial_wave_function_1(x, a, B))



@njit(parallel=True)
def probability_density_2(x, beta, A, trial_wave_function_2):
    
    return np.square(trial_wave_function_2(x, beta, A))



@njit(parallel=True)
def local_energy_1(x, a):

    if np.abs(x) <= a:
        return (1 / (np.square(a) - np.square(x))) + np.square(x) / 2
    
    else:
        return 0



@njit(parallel=True)
def local_energy_2(x, beta):
    return beta + (0.5 - 2 * np.square(beta)) * np.square(x)
