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

def lin(x, m, b):
    return m * x + b
def exp(x, a):
    return np.exp(a * x)

def exp_fit(x, y):
    params_raw, cov = curve_fit(exp, x, y)
    errors = np.sqrt(np.diag(cov))
    return (ufloat(params_raw[0], errors[0]))

def linfit(x, y):
    params_raw, cov = curve_fit(lin, x, y)
    errors = np.sqrt(np.diag(cov))
    return (ufloat(params_raw[0], errors[0]), ufloat(params_raw[1], errors[1]))

#BESTIMMUNG DER SCHALLGESCHWINDIGKEIT MIT DER IMPUS-ECHO METHODE
s, t_1, t_2 = np.genfromtxt('run_time.txt', unpack = True)
delta_t = t_2 - t_1
m, b = linfit(delta_t/2, s)
v_e = Q_(m, 'millimeter/(microsecond)').to('meter/second')
b_e = Q_(b, 'millimeter')
r.app(r'c\ua{E}', v_e)
r.app('b_e', b_e)
t_plot = np.linspace(10, 45, 100)
plt.plot(t_plot, lin(t_plot, m.n, b.n), 'b-', label= 'Lineare Regression')
plt.plot(delta_t/2, s, 'rx', label='Messwerte')
plt.legend(loc='best')
plt.xlabel(r'$\frac{\Delta t}{2}$ / $ \mu $s', fontsize = 12)
plt.ylabel('$s$/mm',  fontsize = 12)
plt.grid()
plt.xlim(t_plot[0], t_plot[-1])
plt.savefig('schallgeschwindigkeit.pdf')
plt.clf()

#SCHALLGESCHWINDIGKEIT MIT DURCHSCHALLUNGSVERFAHREN
s, t_d = np.genfromtxt('run_time_durchschallung.txt', unpack = True)
m, b = linfit(t_d, s)
v_d = Q_(m, 'millimeter/(microsecond)').to('meter/second')
b_d = Q_(b, 'millimeter')
r.app(r'c\ua{D}', v_d)
r.app('b_d', b_d)
t_plot = np.linspace(10, 48, 100)
plt.plot(t_plot, lin(t_plot, m.n, b.n), 'b-', label= 'Lineare Regression')
plt.plot(t_d, s, 'rx', label='Messwerte')
plt.legend(loc='best')
plt.xlabel(r'$t$  / $ \mu $s', fontsize = 12)
plt.ylabel('$s$/mm',  fontsize = 12)
plt.grid()
plt.xlim(t_plot[0], t_plot[-1])
plt.savefig('schallgeschwindigkeit_durchschallung.pdf')
plt.clf()


#BESTIMMUNG DER DÄMPFUNG
L, U_1, U_2 = np.genfromtxt('dämpfung.txt', unpack = True)
plt.plot(L, U_2/U_1, 'rx', label = 'Messwerte')
a = exp_fit(L, U_2/U_1)
r.app(r'\alpha' ,Q_(a, '1/millimeter'))
x_plot = np.linspace(30, 65, 1000)
plt.plot(x_plot, exp(x_plot, a.n), 'b-', label = 'Fit')
plt.xlim(x_plot[0], x_plot[-1])
plt.xlabel('$x$ / mm', fontsize = 12)
plt.grid()
plt.legend(loc = 'best')
plt.ylabel(r'$\frac{U}{U_0}$', fontsize = 14)
plt.savefig('dämpfung.pdf')

#MITTELWERT DER GESCHWINDIGKEITEN
v_mid = np.mean([v_e, v_d])
r.app(r'v\ua{mid}', v_mid)

#PLATTEN
t = np.genfromtxt('platten.txt', unpack = True)
t_1 = Q_(t[1] - t[0], 'microsecond')
t_2 = Q_(t[2] - t[1], 'microsecond')
p_1 = (t_1/2 * v_mid).to('millimeter')
p_2 = (t_2/2 * v_mid).to('millimeter')
r.app('p_1', p_1)
r.app('p_2', p_2)


#AUGE
t = np.genfromtxt('auge.txt', unpack = True)
t_1 = Q_(t[1] - t[0], 'microsecond')
t_2 = Q_(t[2] - t[1], 'microsecond')
t_3 = Q_(t[3] - t[2], 'microsecond')
t_4 = Q_(t[4] - t[3], 'microsecond')
a_1 = (t_1/2 * v_mid).to('millimeter')
a_2 = (t_2/2 * v_mid).to('millimeter')
a_3 = (t_3/2 * v_mid).to('millimeter')
a_4 = (t_4/2 * v_mid).to('millimeter')
r.app('a_1', a_1)
r.app('a_2', a_2)
r.app('a_3', a_3)
r.app('a_4', a_4)















r.makeresults()
