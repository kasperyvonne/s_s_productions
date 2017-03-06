import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
import uncertainties
from pandas import Series, DataFrame
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import scipy.constants as const

U_T_K = np.genfromtxt('krit_temp.txt', unpack=True)
ref_0 = 0.102
ref_m196 = -5.8
m = 196 / (ref_0 - ref_m196)
b = -m * ref_0

def f(x):
    return m * x + b
T_K_theo = -180
T_K_raw = f(U_T_K)
T_K = ufloat(np.mean(T_K_raw), np.std(T_K_raw))
print(T_K)

u = np.linspace(-6, 1)
plt.plot(u, f(u), label='T(U)')
plt.plot([ref_0,ref_m196] , [0, -196], 'rx', label= 'Referenzpunkte')
plt.xlabel('Spannung $U$ in mV')
plt.ylabel('Temperatur T in °C')
plt.grid()
plt.xlim(-6, 0.5)
plt.legend(loc='best')
plt.savefig('temp_skala.pdf')
