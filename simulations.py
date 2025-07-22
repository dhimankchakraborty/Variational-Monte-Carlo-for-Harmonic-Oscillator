import numpy as np
from functions import *
from numba import njit, jit, prange




# @jit(parallel=True)
def simulation_twf_1(mc_step_no, a, x_lim):

    B = np.sqrt(15 / (16 * np.pow(a, 5)))

    rng = np.random.default_rng()

    res = 0
    for i in prange(mc_step_no):
        x = (rng.random() * 2 * x_lim) - x_lim 
        res += probability_density_1(x, a, B, trial_wave_function_1) * local_energy_1(x, a)

    energy_1 = (res / mc_step_no) * 2 * x_lim

    return energy_1



# @jit(parallel=True)
def simulation_twf_2(mc_step_no, beta, x_lim):

    A = np.pow(((2 * beta) / np.pi), 0.25)

    rng = np.random.default_rng()

    res = 0
    for i in prange(mc_step_no):
        x = (rng.random() * 2 * x_lim) - x_lim 
        res += probability_density_2(x, beta, A, trial_wave_function_2) * local_energy_2(x, beta)

    energy_2 = (res / mc_step_no) * 2 * x_lim

    return energy_2