import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
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
N_raw = np.genfromtxt('counts_plateau.txt', unpack=True)
delta_t = 60
errors_raw = np.sqrt(N_raw)
N = unp.uarray(N_raw, errors_raw)/delta_t
U = Q_(np.linspace(310, 700, 40), 'volt')
I = np.genfromtxt('strom.txt', unpack=True)

#l.Latexdocument('tabs/zählrate_strom.tex').tabular([U.magnitude, unp.nominal_values(N), unp.std_devs(N), I],
#'{$U/ \si{\\volt}$} & x & y & {$I/\si{\micro\\ampere}$}', [0, 0, 0, 1],
# caption = 'Zählraten und Ionisationsströme unter verschiedenen Beschleunigungsspannungen $U$.',
# label='zaelrate_strom')



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
N_1 = ufloat(25692, np.sqrt(25692))/60
N_2 = ufloat(1109, np.sqrt(1109))/60
N_12 = ufloat(26775, np.sqrt(26775))/60
T_zwei = totzeit(N_1, N_2, N_12)
r.app('T_2', Q_(T_zwei, 'second').to('microsecond'))


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
plt.ylabel('Zählrate in 1/s', fontsize = 11)
plt.xlabel('Spannung in V', fontsize = 11)
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

r.makeresults()
