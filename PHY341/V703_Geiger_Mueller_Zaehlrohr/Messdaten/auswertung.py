import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import random
import latex as l
u = UnitRegistry()
Q_ = u.Quantity
r = l.Latexdocument('results.tex')
###########################################
#DATEN EINLESEN
#x = np.genfromtxt('bla.txt', unpack=True)

###########################################
#FUNKTIONEN

def lin(x, m, b):
    return m * x + b

def linfit(x, y):
    params_raw, cov = curve_fit(lin, x, y)
    errors = np.sqrt(np.diag(cov))
    return (ufloat(params_raw[0], errors[0]), ufloat(params_raw[1], errors[1]))

def totzeit(n1, n2, n12):
    return (n1 + n2 - n12) / (2 * n1 * n2)

###########################################
#BERECHNUNGEN
x = np.linspace(0, 20, 50)
y = np.array([5 * i + 3 + random.randrange(-10, 10, 1)/5 for i in x])
a, b = linfit(x, y)
r.app('a', Q_(a, 'volt/second'))
r.app('b', Q_(b, 'volt'))




###########################################
#PLOTS

t = np.linspace(-1, 21)
plt.plot(t, lin(t, a.n, b.n), 'r-', label = 'Fit')
plt.errorbar(x, y, yerr = np.random.rand(2, len(y)) * 20, fmt='bx', ecolor = 'b',
elinewidth = 1, capsize = 2, label='Values')
plt.xlabel('Zeit in s', fontsize = 11)
plt.ylabel('Spannung in V', fontsize = 11)
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/test.pdf')
r.makeresults()
