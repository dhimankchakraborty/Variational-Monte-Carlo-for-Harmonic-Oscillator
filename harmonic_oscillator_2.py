import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from functions import *
from simulations import *




mc_step_no = 1000000
# therm_step_no = mc_step_no

beta_arr = np.round(np.linspace(0, 2, 21)[1:], 3)
x_lim = 5

# beta = 0.5

energy_2_arr = np.zeros((len(beta_arr)))

for j, beta in enumerate(beta_arr):


    energy_2_arr[j] = simulation_twf_2(mc_step_no, beta, x_lim)

    print(f'{j} | {beta} | {energy_2_arr[j]}')

# print(energy_2)
# print(beta_arr)

plt.plot(beta_arr, energy_2_arr, marker='o')
plt.title(r'Energy vs Parameter ($\beta$) $[\Psi_t = Ae^{-\beta x^2}]$')
plt.xlabel(r'Parameter ($\beta$)')
plt.ylabel(r'Energy ($\epsilon$)')
plt.grid()
plt.show()