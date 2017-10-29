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


#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS



#FITFUNKTIONEN
def hysterese(I, a, b, c, d):
    return a * I**3 + b * I**2 + c * I + d




#######LOAD DATA#########
B_up, B_down = np.genfromtxt('data/hysterese.txt', unpack=True) #Flussdichte bei auf und absteigendem Spulenstrom
I = np.linspace(0, 20, 21) # Stromstärke von 0 bis 20A in 1A Schritten
l.Latexdocument('tabs/hysterese.tex').tabular(
data = [I, B_up, B_down], #Data incl. unpuarray
header = [r'I / \ampere', r'B\ua{auf} / \milli\tesla', r'B\ua{ab} / \milli\tesla'],
places = [1, 0, 0],
caption = r'Gemessene magnetische Flussdichten $B\ua{i}$ bei auf- bzw. absteigenden Strom $I$.',
label = 'hysterese')

###FIT DER HYSTERESE######
params_up, cov_up = curve_fit(hysterese, I, B_up)
params_down, cov_down = curve_fit(hysterese, I, B_down)







#######PLOTS#########
plt.plot(I, B_up, 'rx', label = 'Messwerte')
I_plot = np.linspace(-1, 21, 100)
plt.plot(I_plot, hysterese(I_plot, *params_up), label = 'Fit')
plt.xlim(-1, 21)
plt.xlabel('$I/$A')
plt.ylabel('$B/$mT')
plt.legend()
plt.grid()
plt.savefig('plots/hysterese_aufsteigend.pdf')

plt.clf()
plt.plot(I, B_down, 'rx', label = 'Messwerte')
plt.plot(I_plot, hysterese(I_plot, *params_down), label = 'Fit')
plt.xlim(-1, 21)
plt.xlabel('$I/$A')
plt.ylabel('$B/$mT')
plt.legend()
plt.grid()
plt.savefig('plots/hysterese_absteigend.pdf')

plt.clf()
plt.plot(I, B_up, 'rx', label = 'Messwerte aufsteigender Strom')
plt.plot(I, B_down, 'bx', label = 'Messwerte absteigender Strom')
plt.xlim(-1, 21)
plt.xlabel('$I/$A')
plt.ylabel('$B/$mT')
plt.legend()
plt.grid()
plt.savefig('plots/hysterese_data.pdf')
plt.clf()
params_up = correlated_values(params_up, cov_up)
params_down = correlated_values(params_down, cov_down)
print(params_up)
print(params_down)



#Rechnungen mit den Positionen der Aufspaltung
peaks_rot_0 = np.genfromtxt('data/peaks_rot_sigma_I_0.txt', unpack = True)
peaks_rot_10 = np.genfromtxt('data/peaks_rot_sigma_I_10.txt', unpack = True)
#l.Latexdocument('tabs/peaks_rot.tex').tabular(
#data = [peaks_rot_0, unp.uarray(peaks_rot_10[:12], peaks_rot_10[12:])],
#header = ['x_0 / px', 'x_{10} / px'],
#places = [0, (4.0, 4.0)],
#caption = r'Positionen $x_0$ und $x_{10}$ der Intensitätsmaxima unter $I= \SI{0}{\ampere}$ und $I= \SI{10}{\ampere}$.',
#label = 'tab: peaks_rot')

delta_s_rot = peaks_rot_0[1:] - peaks_rot_0[:-1]
del_s_rot = (peaks_rot_10[1:] - peaks_rot_10[:-1])[::2]
del_s_mid = [(del_s_rot[i] + del_s_rot[i+1])/2 for i in range(0, len(del_s_rot)-1)]





lambda_rot = Q_(643.8, 'nanometer').to('meter')
d = Q_(4, 'millimeter').to('meter')
c = Q_(const.c, 'meter / second')
h = Q_(const.h, 'joule * second')
mu_B = Q_(const.physical_constants['Bohr magneton'][0], 'joule / tesla')
n_rot = 1.4567
del_lambda_rot = (1/2 * lambda_rot**2 / (2 * d * np.sqrt(n_rot**2 - 1) ) * (del_s_mid / delta_s_rot)).to('picometer')
delta_E_rot = (h * c / lambda_rot**2 * del_lambda_rot).to('eV')
g_rot = (delta_E_rot / (mu_B * Q_(hysterese(10, *params_up), 'millitesla'))).to('dimensionless')
g_rot_mid = np.mean(g_rot)
g_rot = unp.uarray(noms(g_rot), stds(g_rot))
print('Mittelwert Lande-Faktor: ', g_rot_mid)
print('B-Feldstärken: ', hysterese(0, *params_up), hysterese(10, *params_up))

#l.Latexdocument('tabs/abstände_rot.tex').tabular(
#data = [delta_s_rot, del_s_mid, del_lambda_rot.magnitude, delta_E_rot.magnitude*1e5, g_rot], #Data incl. unpuarray
#header = [r'\Delta s_i / px', r'\frac{\delta s_i + \delta s_{i+1}}{2} / px', r'\Delta \lambda / \pico\meter',
#        r'\Delta E / e-5\electronvolt', 'g / '],
#places = [0, 0, 1, 1, (1.3, 1.3)],
#caption = r'Abstände zwichen den unaufgespaltenen roten Linien $\Delta s_i$ und gemittelte Abstände \frac{\delta s_i + \delta s_{i+1}}{2}. Wellenlängenverschiebung $\Delta \lambda$, '
# 'Energieaufspaltung $\Delta E$ und berechnete Übergangs-Lande-Faktoren g.',
#label = 'hysterese')

r.makeresults()
