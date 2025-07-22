import numpy as np
from functions import *
from numba import njit, jit, prange




# @jit(parallel=True)
def simulation_twf_1(mc_step_no, a, x_lim):

    B = np.sqrt(15 / (16 * np.pow(a, 5)))

    rng = np.random.default_rng()

    res = 0
    res2 = 0
    for i in prange(mc_step_no):
        x = (rng.random() * 2 * x_lim) - x_lim 
        Edx = probability_density_1(x, a, B, trial_wave_function_1) * local_energy_1(x, a)
        res += Edx
        res2 += np.square(Edx)

    energy = (res / mc_step_no) * 2 * x_lim
    variance = (res2 / mc_step_no) * 2 * x_lim - np.square(energy)

    return energy, variance



# @jit(parallel=True)
def simulation_twf_2(mc_step_no, beta, x_lim):

    A = np.pow(((2 * beta) / np.pi), 0.25)

    rng = np.random.default_rng()

    res = 0
    res2 = 0
    for i in prange(mc_step_no):
        x = (rng.random() * 2 * x_lim) - x_lim 
        Edx = probability_density_2(x, beta, A, trial_wave_function_2) * local_energy_2(x, beta)
        res += Edx
        res2 += np.square(Edx)

    energy = (res / mc_step_no) * 2 * x_lim
    variance = np.abs((res2 / mc_step_no) - np.square(energy))

    return energy, variance / mc_step_no # np.sqrt(np.abs(variance))