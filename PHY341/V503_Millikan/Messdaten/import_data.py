import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
from scipy.constants import physical_constants as const
from fractions import gcd
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#NATURKONSTANTEN
e_0 = Q_(const['elementary charge'][0], 'coulomb')
g = Q_(const['standard acceleration of gravity'][0], 'meter/(second**2)')
N_A = Q_(const['Avogadro constant'][0], '1/mole')
r.app(r'e\ua{0}', e_0)
r.app(r'g', g)
r.app(r'N\ua{A}', N_A   )

time_raw, voltage_raw, resistance_raw = np.genfromtxt('schwebe_methode.txt', unpack=True)

time = Q_(time_raw, 'second')
voltage = Q_(voltage_raw, 'volt')
resistance = Q_(resistance_raw, 'megaohm')


T_fit = np.linspace(20, 39, 20)
R_fit = np.genfromtxt('temp.txt', unpack=True)
coeff = np.polyfit(R_fit, T_fit, deg = len(T_fit))
plt.plot(R_fit, T_fit, 'rx')
R_plot = np.linspace(1.470, 2.310, 1000)

plt.plot(R_plot, np.polynomial.polynomial.polyval(R_plot, coeff[::-1]))
plt.xlim(R_plot[0], R_plot[-1])
plt.savefig('temperature_fit.pdf')


def mid(x):
    return ufloat(np.mean(x), 1/np.sqrt(len(x)) * np.std(x)  )
def temp(R):
    return Q_(np.polynomial.polynomial.polyval(R, coeff[::-1]), 'celsius')

eta_1 = 1.85
eta_2 = 1.88
T_1 = 16
T_2 = 32
m = (eta_1 - eta_2)/(T_1 - T_2)
b = 0.5 * (eta_1 + eta_2 - m *(T_1 + T_2))
def viskositaet(T):
    return Q_( m * T + b, 'newton * second/(meter**2)' )

print(m, b)    
