import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functions import *




mc_step_no = 100000
# therm_step_no = mc_step_no

beta_arr = np.round(np.linspace(0, 2, 21)[1:], 3)
x_lim = 5

energy_1_arr = np.zeros((len(beta_arr)))

for j, beta in enumerate(beta_arr):

    A = np.pow(((2 * beta) / np.pi), 0.25)

    rng = np.random.default_rng()

    res = 0
    for i in prange(mc_step_no):
        x = (rng.random() * 2 * x_lim) - x_lim 
        res += probability_density_1(x, a, B, trial_wave_function_1) * local_energy_2(x, beta)

    energy_2 = res / mc_step_no

    energy_1_arr[j] = energy_2

    print(f'{j} | {beta} | {energy_2}')