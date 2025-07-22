import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functions import *
from simulations import *




mc_step_no = 1000000
# therm_step_no = mc_step_no

# a = 2
# B = np.sqrt(15 / (16 * np.pow(a, 5)))
a_arr = np.round(np.linspace(1, 3, 21)[1:], 3)

x_lim = 5

energy_1_arr = np.zeros((len(a_arr)))

for j, a in enumerate(a_arr):


    energy_1_arr[j] = simulation_twf_1(mc_step_no, a, x_lim)

    print(f'{j} | {a} | {energy_1_arr[j]}')

# print(energy_2)
# print(beta_arr)

plt.plot(a_arr, energy_1_arr, marker='o')
plt.title(r'Energy vs Parameter ($a$) [$\Psi_t = (a^2 - x^2)$ for $|x| \leq a$, 0 elsewhere]')
plt.xlabel(r'Parameter ($a$)')
plt.ylabel(r'Energy ($\epsilon$)')
plt.grid()
plt.show()