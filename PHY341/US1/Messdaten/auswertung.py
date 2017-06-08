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

#BESTIMMUNG DER SCHALLGESCHWINDIGKEIT MIT DER IMPUS-ECHO METHODE, BEZEICHNER _e
s_e, t_1_e, t_2_e = np.genfromtxt('data/run_time.txt', unpack = True)
delta_t = t_2_e - t_1_e
m_e, b_e = linfit(delta_t/2, s_e)
v_e = Q_(m_e, 'millimeter/(microsecond)').to('meter/second')
z_e = Q_(b_e, 'millimeter')
r.app(r'c\ua{E}', v_e)
r.app('b_e', z_e)
t_plot = np.linspace(10, 45, 100)
plt.plot(t_plot, lin(t_plot, m_e.n, b_e.n), 'b-', label= 'Lineare Regression')
plt.plot(delta_t/2, s_e, 'rx', label='Messwerte')
plt.legend(loc='best')
plt.xlabel(r'$\frac{\Delta t}{2}$ / $ \mu $s', fontsize = 12)
plt.ylabel('$s$/mm',  fontsize = 12)
plt.grid()
plt.xlim(t_plot[0], t_plot[-1])
plt.savefig('plots/schallgeschwindigkeit.pdf')
plt.clf()

#SCHALLGESCHWINDIGKEIT MIT DURCHSCHALLUNGSVERFAHREN
s_d, t_d = np.genfromtxt('data/run_time_durchschallung.txt', unpack = True)
m_d, b_d = linfit(t_d, s_d)
v_d = Q_(m_d, 'millimeter/(microsecond)').to('meter/second')
z_d = Q_(b_d, 'millimeter')
r.app(r'c\ua{D}', v_d)
r.app('b_d', z_d)
t_plot = np.linspace(10, 48, 100)
plt.plot(t_plot, lin(t_plot, m_d.n, b_d.n), 'b-', label= 'Lineare Regression')
plt.plot(t_d, s_d, 'rx', label='Messwerte')
plt.legend(loc='best')
plt.xlabel(r'$t$  / $ \mu $s', fontsize = 12)
plt.ylabel('$s$/mm',  fontsize = 12)
plt.grid()
plt.xlim(t_plot[0], t_plot[-1])
plt.savefig('plots/schallgeschwindigkeit_durchschallung.pdf')
plt.clf()


#BESTIMMUNG DER DÄMPFUNG
L, U_1, U_2 = np.genfromtxt('data/dämpfung.txt', unpack = True)
plt.plot(L, U_2/U_1, 'rx', label = 'Messwerte')
#DÄMPFUNGSFAKTOR a
a = exp_fit(L, U_2/U_1)
r.app(r'\alpha' ,-Q_(a, '1/millimeter'))

x_plot = np.linspace(30, 65, 1000)
plt.plot(x_plot, exp(x_plot, a.n), 'b-', label = 'Interpolation')
plt.xlim(x_plot[0], x_plot[-1])
plt.xlabel('$x$ / mm', fontsize = 12)
plt.grid()
plt.legend(loc = 'best')
plt.ylabel(r'$\frac{U_2}{U_1}$', fontsize = 14)
plt.savefig('plots/dämpfung.pdf')

#MITTELWERT DER GESCHWINDIGKEITEN
v_theo = Q_(2730, 'meter/second')
v_mid = np.mean([v_e, v_d])
b_mid = np.mean([z_e, z_d])
r.app(r'b\ua{mid}', b_mid)

r.app(r'v\ua{mid}', v_mid)
r.app(r'd\ua{v}', v_mid/v_theo - 1)

#PLATTEN
t_p = np.genfromtxt('data/platten.txt', unpack = True)
t_1_p = Q_(t_p[1] - t_p[0], 'microsecond')
t_2_p = Q_(t_p[2] - t_p[1], 'microsecond')
p_1 = (lin(t_1_p/2, v_e, z_e)).to('millimeter')
p_2 = (lin(t_2_p/2, v_e, z_e)).to('millimeter')
r.app('p_1', p_1)
r.app('p_2', p_2)


#AUGE
v_L = Q_(2500, 'meter/second')
v_GK = Q_(1410, 'meter/second')
t_a = np.genfromtxt('data/auge.txt', unpack = True)
t_1_a = Q_(t_a[1] - t_a[0], 'microsecond')
t_2_a = Q_(t_a[2] - t_a[1], 'microsecond')
t_3_a = Q_(t_a[3] - t_a[2], 'microsecond')
t_4_a = Q_(t_a[4] - t_a[3], 'microsecond')
a_1 = (t_1_a/2 * v_GK).to('millimeter')
a_2 = (t_2_a/2 * v_GK).to('millimeter')
a_3 = (t_3_a/2 * v_L).to('millimeter')
a_4 = (t_4_a/2 * v_GK).to('millimeter')
l_exp = a_1 + a_2 + a_3 + a_4
r.app(r'l\ua{exp}', l_exp/3)
r.app('a_1', a_1)
r.app('a_2', a_2)
r.app('a_3', a_3)
r.app('a_4', a_4)



#CEPSTRUM
x, y = np.genfromtxt('data/cepstrum_data.txt', unpack = True)
plt.clf()
plt.plot(x[x > 26], y[x > 26], 'k-', label = 'Messdaten', linewidth = 1.5)
plt.axvline(x = t_p[0], ls='--', color='r', label = 'Abgelesene Zeitdifferenzen')
plt.axvline(x = t_p[1], ls='--', color='r')
plt.axvline(x = t_p[2], ls='--', color='r')
plt.xlabel('$t/\mu$s')
plt.ylabel('$U/$V')
plt.legend(loc='best')
plt.grid()
plt.xlim(26, 50)
plt.savefig('plots/cepstrum.pdf')


#SPECTRUM
x, y = np.genfromtxt('data/spectrum_data.txt', unpack = True)
plt.clf()
plt.plot(x, y, 'k-', label = 'Messdaten', linewidth = 0.5)
plt.xlabel(r'$\nu/$MHz')
plt.ylabel('FFT')
plt.legend(loc='best')
plt.grid()
plt.xlim(0, 20)
plt.savefig('plots/spectrum.pdf')


#AUGE
x, y = np.genfromtxt('data/auge_data.txt', unpack = True)
plt.clf()
plt.plot(x, y, 'k-', label = 'Messdaten')
plt.axvline(x = t_a[0], ls='--', color='r', label = 'Abgelesene Zeitdifferenzen')
plt.axvline(x = t_a[1], ls='--', color='r')
plt.axvline(x = t_a[2], ls='--', color='r')
plt.axvline(x = t_a[3], ls='--', color='r')
plt.axvline(x = t_a[4], ls='--', color='r')
plt.xlabel('$t/\mu$s')
plt.ylabel('$U/$V')
plt.legend(loc='best')
plt.grid()
plt.xlim(0, 80)
plt.savefig('plots/auge.pdf')





r.makeresults()
