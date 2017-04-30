import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import scipy.constants
import matplotlib.pyplot as plt
from pint import UnitRegistry
import random
import latex as l
u = UnitRegistry()
Q_ = u.Quantity
r = l.Latexdocument('results.tex')
e = Q_(scipy.constants.e, 'coulomb')
###########################################
#DATEN EINLESEN
N_raw = np.genfromtxt('counts_plateau.txt', unpack=True)
delta_t = 60
errors_raw = np.sqrt(N_raw)
N = unp.uarray(N_raw, errors_raw)/delta_t
U = Q_(np.linspace(310, 700, 40), 'volt')
I = np.genfromtxt('strom.txt', unpack=True)
#Ladungsmenge
delta_Q = Q_(I/N, 'microampere * second').to('gigacoulomb') / e


l.Latexdocument('tabs/zählrate_strom_ladung.tex').tabular([U.magnitude, N_raw, np.sqrt(N_raw), unp.nominal_values(N), unp.std_devs(N), I, unp.nominal_values(delta_Q), unp.std_devs(delta_Q)],
'{$U/ \si{\\volt}$} & x & y & {$I/\si{\micro\\ampere}$}', [2, 2, 2, 2, 2, 2, 3, 2],
 caption = 'Gemessene Impulszahlen $Z$ und Ionisationsströme $I$ unter verschiedenen Beschleunigungsspannungen $U$ und berechnete Zählraten $N$ und pro einfallendem Teilchen freigesetzte Ladungsmenge $Q$',
 label='zaelrate_strom')



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
#Plateau
U_plateau = U.magnitude[6:30]
N_plateau = N[6:30]
m_plateau, offset_plateau = linfit(U_plateau, unp.nominal_values(N_plateau))
r.app('m', Q_(m_plateau, '1/(volt*second)'))
r.app('b', Q_(offset_plateau, '1/second'))

#totzeit, direkte Messung
U_T_1, T_1 = np.genfromtxt('totzeit_oszi.txt', unpack=True)
T_oszi = ufloat(np.mean(T_1), 1/np.sqrt(5) * np.std(T_1))
r.app('T_1', Q_(T_oszi, 'microsecond'))
l.Latexdocument('tabs/totzeit_oszi.tex').tabular([U_T_1, T_1],
'{$U/ \si{\\volt}$} & {$T/ \si{\\micro\second}$}', [0, 0],
 caption = 'Mit dem Oszilloskop gemessene Totzeiten $T$ unter verschiedenen Beschleunigungsspannungen $U$.',
 label='totzeit_oszi')

#Totzeit über Zweier-Methode
Z_1 = ufloat(25692, np.sqrt(25692))
Z_2 = ufloat(1109, np.sqrt(1109))
Z_12 = ufloat(26775, np.sqrt(26775))
N_1 = Z_1/60
N_2 = Z_2/60
N_12 = Z_12/60
r.app('Z_1', Q_(Z_1, 'dimensionless'))
r.app('Z_2', Q_(Z_2, 'dimensionless'))
r.app('Z_{1+2}', Q_(Z_12, 'dimensionless'))
r.app('N_1', Q_(N_1, '1/second'))
r.app('N_2', Q_(N_2, '1/second'))
r.app('N_{1+2}', Q_(N_12, '1/second'))
T_zwei = totzeit(N_1, N_2, N_12)
r.app('T_2', Q_(T_zwei, 'second').to('microsecond'))
print(1- T_zwei/T_oszi)

#Erholungszeit
U_T_E, T_E = np.genfromtxt('erholungszeit.txt', unpack=True)
T_erholung = ufloat(np.mean(T_E), 1/np.sqrt(5) * np.std(T_E))
l.Latexdocument('tabs/erholungszeit.tex')
r.app('T_E', Q_(T_erholung, 'microsecond'))
l.Latexdocument('tabs/erholungszeit.tex').tabular([U_T_E, T_E],
'{$U/ \si{\\volt}$} & {$T_E/ \si{\\micro\second}$}', [0, 0],
 caption = 'Mit dem Oszilloskop gemessene Erholungszeiten $T_E$ unter verschiedenen Beschleunigungsspannungen $U$.',
 label='erholungszeit')





###########################################
#PLOTS

plt.errorbar(U.magnitude, unp.nominal_values(N), yerr = unp.std_devs(N), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
plt.errorbar((U.magnitude[6], U.magnitude[29]), (unp.nominal_values(N)[6],unp.nominal_values(N)[29]) , yerr = (unp.std_devs(N)[6],unp.std_devs(N)[29]) , fmt='rx',
ecolor = 'r', elinewidth = 1, capsize = 2, label='Gewählte Schranken Plateaubereich')
plt.ylabel('Zählrate $N$ in 1/s', fontsize = 11)
plt.xlabel('Spannung $U$ in V', fontsize = 11)
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/all_counts.pdf')



#PLATEAU
plt.clf()
U_plot = np.linspace(360, 610, 100)
plt.plot(U_plot, lin(U_plot, m_plateau.n, offset_plateau.n), 'r-', label='Regressionsgerade')
plt.errorbar(U_plateau, unp.nominal_values(N_plateau), yerr = unp.std_devs(N_plateau), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
plt.ylabel('Zählrate $N$ in 1/s', fontsize = 11)
plt.xlabel('Spannung $U$ in V', fontsize = 11)
plt.xlim(U_plot[0], U_plot[-1])
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/plateau.pdf')

plt.clf()
Q = np.array(delta_Q.magnitude)

plt.errorbar(U.magnitude, unp.nominal_values(Q), yerr = unp.std_devs(Q), fmt='bx', ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
plt.xlabel('Spannung $U$ in V', fontsize = 11)
plt.ylabel('Ladungsmenge $\Delta Q$ in Ge', fontsize = 11)
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/ladung.pdf')

r.makeresults()
