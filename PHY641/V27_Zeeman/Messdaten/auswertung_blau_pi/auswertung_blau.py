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
#import pandas as pd
#from pandas import Series, DataFrame
import matplotlib.image as mpimg
import scipy.constants as const
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})


#FITFUNKTIONEN
def hysterese(I, a, b, c, d):
    return a * I**3 + b * I**2 + c * I + d




#######LOAD DATA#########
B_up, B_down = np.genfromtxt('../data/hysterese.txt', unpack=True) #Flussdichte bei auf und absteigendem Spulenstrom
I = np.linspace(0, 20, 21) # Stromstärke von 0 bis 20A in 1A Schritten
#l.Latexdocument('tabs/hysterese.tex').tabular(
#data = [I, B_up, B_down], #Data incl. unpuarray
#header = [r'I / \ampere', r'B\ua{auf} / \milli\tesla', r'B\ua{ab} / \milli\tesla'],
#places = [1, 0, 0],
#caption = r'Gemessene magnetische Flussdichten $B\ua{i}$ bei auf- bzw. absteigenden Strom $I$.',
#label = 'hysterese')

###FIT DER HYSTERESE######
params_up, cov_up = curve_fit(hysterese, I, B_up)
params_down, cov_down = curve_fit(hysterese, I, B_down)


params_up = correlated_values(params_up, cov_up)
params_down = correlated_values(params_down, cov_down)




#Rechnungen mit den Positionen der Aufspaltung
peaks_blau_0 = np.genfromtxt('../data/peaks_blau_pi_I_0.txt', unpack = True)
peaks_blau_17 = np.genfromtxt('../data/peaks_blau_pi_I_17.txt', unpack = True)
#l.Latexdocument('../tabs/peaks_blau_pi.tex').tabular(
#data = [peaks_blau_0, unp.uarray(peaks_blau_17[:10], peaks_blau_17[10:])],
#header = ['x_0 / px', 'x_{17} / px'],
#places = [0, (4.0, 4.0)],
#caption = r'Blaue Pi Aufspaltung: Positionen $x_0$ und $x_{17}$ der Intensitätsmaxima unter $I= \SI{0}{\ampere}$ und $I= \SI{17}{\ampere}$.',
#label = 'tab: peaks_blau_pi')

delta_s_blau = peaks_blau_0[1:] - peaks_blau_0[:-1]
del_s_blau = (peaks_blau_17[1:] - peaks_blau_17[:-1])[::2]
del_s_mid = [(del_s_blau[i] + del_s_blau[i+1])/2 for i in range(0, len(del_s_blau)-1)]



lambda_blau = Q_(480.0, 'nanometer').to('meter')
d = Q_(4, 'millimeter').to('meter')
c = Q_(const.c, 'meter / second')
h = Q_(const.h, 'joule * second')
mu_B = Q_(const.physical_constants['Bohr magneton'][0], 'joule / tesla')
n_blau = 1.41735
del_lambda_blau = (1/2 * lambda_blau**2 / (2 * d * np.sqrt(n_blau**2 - 1) ) * (del_s_mid / delta_s_blau)).to('picometer')
delta_E_blau = (h * c / lambda_blau**2 * del_lambda_blau).to('eV')
g_blau = (delta_E_blau / (mu_B * Q_(hysterese(17, *params_up), 'millitesla'))).to('dimensionless')
g_blau_mid = np.mean(g_blau)
g_blau = unp.uarray(noms(g_blau), stds(g_blau))
print(g_blau_mid)

#l.Latexdocument('../tabs/abstände_blau_pi.tex').tabular(
#data = [delta_s_blau, del_s_mid, del_lambda_blau.magnitude, delta_E_blau.magnitude*1e5, g_blau], #Data incl. unpuarray
#header = [r'\Delta s_i / px', r'\frac{\delta s_i + \delta s_{i+1}}{2} / px', r'\Delta \lambda / \pico\meter',
#        r'\Delta E / 10^{-5}\electronvolt', 'g / '],
#places = [0, 0, 1, 1, (1.3, 1.3)],
#caption = r'Blaue Pi Aufspaltung: Abstände zwischen den unaufgespaltenen Linien $\Delta s_i$ und gemittelte Abstände \frac{\delta s_i + \delta s_{i+1}}{2}. Wellenlängenverschiebung $\Delta \lambda$, '
# + 'Energieaufspaltung $\Delta E$ und berechneter Übergangs-Landé-Faktor g.',
#label = 'abstände_blau_pi')
