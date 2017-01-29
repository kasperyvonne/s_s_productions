import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex
u = UnitRegistry()
Q_ = u.Quantity
#Apparaturkonstanten
R = Q_(1, 'ohm')
L = Q_(1, 'henry')
C = Q_(1, 'farad')

def exp_function(t, A, B, C):
    return A * np.exp(B * t) + C
x = np.linspace(0, 2, 100)
#Fits
params_dämpfung_raw, covariance_dämpfung = curve_fit(exp_function, x, np.exp(x))
params_dämpfung = unp.uarray(params_dämpfung_raw, np.sqrt(np.diag(covariance_dämpfung)))

#Theoriewerte
R_ap_theo = np.sqrt(4*L/C).to('ohm')
T_ex_theo = (2 * L / R).to('second')

#Plots
#x = np.linspace(0, 2, 1000)
#plt.plot(x, np.exp(x))
#plt.yscale('log')
#plt.show()
