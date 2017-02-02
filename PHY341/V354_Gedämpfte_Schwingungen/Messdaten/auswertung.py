import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
from uncertainties.umath import *
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex
u = UnitRegistry()
Q_ = u.Quantity
#Apparaturkonstanten
R = ufloat(509.5, 0.5)
L = ufloat(10.11e-3, 0.03e-3)
C = ufloat(2.098e-9, 0.006e-9)


#Messwerte einlesen
frequenz1, U_G_1, U_C_1, a1, b1 = np.genfromtxt('frequenzabhängigkeit_amplitude_phase.txt', unpack=True)
frequenz2, U_G_2, U_C_2, a2, b2 = np.genfromtxt('frequenz_2.txt', unpack=True)
frequenz = np.concatenate((frequenz1, frequenz2), axis=0)
U_G = np.concatenate((U_G_1, U_G_2), axis=0)
U_C = np.concatenate((U_C_1, U_C_2), axis=0)
a = np.concatenate((a1, a2), axis=0)
b = np.concatenate((b1, b2), axis=0)
phase = a/b * 2*np.pi
latex.Latexdocument('frequenzabhängigkeit.tex').tabular([frequenz, U_G, U_C, a, b, phase],
'{$\\nu$ / \si{\hertz}} & {$U$ / $\si{\\volt}$} & {$U_C$ / $\si{\\volt}$} & {$a$ / $\si{\micro\second}$} & {$b$ / $\si{\micro\second}$} & {$\phi$ / rad}', [0, 2, 2, 2, 2, 2],
caption = 'Messwerte der Zeitparameter a und b, Kondensator- und Generatorspannung, sowie die errechnete Phasendifferenz in Abhängigkeit von der Frequenz $\\nu$.', label = 'tab: frequenzabhängigkeit')


amplitude = np.genfromtxt('amplitude_schwingfall.txt', unpack=True)
time = np.linspace(0, 12, 13) * 30e-6
latex.Latexdocument('amplitude.tex').tabular([time*100000, amplitude],
'{$t$ / $10^{-5}\si{\second}$} & {$A$ / $\si{\\volt}$}', [1, 2],
caption = 'Zeitlicher Verlauf der Amplitude des gedämpften Schwingkreises', label = 'tab: amplitude')


def lin_function(t, m, b):
    return m * t + b

def exp_function(t, A, B, C):
    return A * np.exp(B*t) + C
#x = np.linspace(0, 2, 100)
##Fits
#params_raw, cov = curve_fit(lin_function, time, np.log(amplitude))
params2_raw, cov2 = curve_fit(exp_function, time, amplitude)
params2 = unp.uarray(params2_raw, np.sqrt(np.diag(cov2)))
print('Parameter der linearen Rgression: ', params2)
print('A_0 = ', params2[0])
mu = -1/(2*np.pi) * params2[0]
#print(mu)
R_eff = -(params2[1]) * 2 * L.n
print('Effektivwiderstand: ', R_eff)

#breite
#nu_plus_theo = -R/(2*L) + sqrt(R**2/(4*L**2) + 1/(L*C))
nu_plus_exp = 37.45e3
nu_minus_exp = 28.6e3
breite_exp = nu_plus_exp - nu_minus_exp
breite_theo = (1/(2*np.pi) * R/L)
d_breite = (breite_exp/breite_theo - 1) * 100
print('Breite exp, theo, proz: ', breite_exp, breite_theo, d_breite)

#breite aus Phase
nu_plus_exp_p = 36500
nu_minus_exp_p = 30000
breite_exp_p = nu_plus_exp_p - nu_minus_exp_p
breite_theo = (1/(2*np.pi) * R/L)
d_breite_p = (breite_exp_p/breite_theo - 1) * 100
print('Breite exp, theo, proz: ', breite_exp_p, breite_theo, d_breite_p)


nu0_exp = 33000
nu0_theo = (1/(2*np.pi) * sqrt(1/(L*C) - R**2/(2*L**2) ))
d_nu0 = (nu0_exp/nu0_theo -1)*100
print('Resonanzfrequenz exp, theo, proz: ', nu0_exp, nu0_theo, d_nu0)
print('Abweichung Resonanzfrequenz aus Phase: ', (34000/nu0_theo -1)*100)
print('Güte', nu0_exp/breite_exp)
print('theoretische überhöhung: ', nu0_theo/breite_theo)

R_ap_theo = sqrt(4 * L/C)
R_ap_exp = ufloat(28, 1) * 1e3
d_R_ap = (R_ap_exp/R_ap_theo -1)*100
print('R_ap exp, theo, proz: ', R_ap_exp, R_ap_theo, d_R_ap)
T_ex_theo = (2 * L / R)
T_ex_exp = 1/params2[1]
print('Abklingzeit, theo, exp, d: ', T_ex_theo, T_ex_exp, (T_ex_exp/T_ex_theo - 1)*100 )
#Plots
#x = np.linspace(0, 2, 1000)
#plt.plot(x, np.exp(x))
#plt.yscale('log')
#plt.show()

def phase_theo(f):
    w = 2 * np.pi * f
    return np.arctan( (-w * (R * C).n ) / (1 - (L * C).n * w**2 ) )

print(U_C[9:]/U_G[9:])
f = np.linspace(24000, 46000, 1000)
u = np.linspace(1,4, 100)
plt.plot(frequenz[9:]/1000, U_C[9:]/U_G[9:], 'rx', label='Messwerte')
plt.plot(f/1000, (np.ones(len(f))* 1/np.sqrt(2) * 17/4.32), '--b', label = 'Niveaulinie $\\frac{1}{\sqrt{2}}\\frac{U_{C, max}}{U}$')
plt.plot(np.ones(len(u))* nu_minus_exp/1000, u, '--g', label = 'Abgelesene Frequenzen $\\nu_{\pm}$')
plt.plot(np.ones(len(u))* nu_plus_exp/1000, u, '--g')
plt.xlim (24.5, 45.5)
plt.legend(loc='best')
plt.xlabel('Frequenz $\\nu$ in kHz')
plt.ylabel('$\\frac{U_C}{U}$')
plt.grid()
plt.savefig('U_f_linear.pdf')

plt.clf()
plt.plot(frequenz[9:]/1000, U_C[9:]/U_G[9:], 'rx', label='Messwerte')
plt.xscale('log')
plt.xlim(frequenz[9:][0]/1000, frequenz[9:][-1]/1000)
#plt.show()

lin = np.linspace(24, 46, 100)
pha = np.linspace(0, np.pi, 100)
plt.clf()
plt.plot(frequenz[9:]/1000, phase[9:], 'rx', label='Messwerte')
plt.yticks([0, np.pi/4, np.pi / 2, 3*np.pi/4 , np.pi],
           [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"], fontsize = 16)
plt.xlim(24.5, 45.5)
plt.plot(lin, np.ones(len(lin))* np.pi/4, '--b')
plt.plot(np.ones(len(pha)) * 30.0, pha, '--g', label='Abgelesene Frequenzen $\\nu_{\pm}$')
plt.plot(lin, np.ones(len(lin))* 3*np.pi/4, '--b')
plt.plot(np.ones(len(pha)) * 36.5, pha, '--g')
plt.grid()
plt.legend(loc='best')
plt.ylabel('Phasenverschiebung $\\varphi$ in rad')
plt.xlabel('Frequenz $\\nu$ in kHz')
#plt.yscale('log')
plt.savefig('phase_f_linear.pdf')


t = np.linspace(time[0]-0.00001, time[-1]+0.00001)
plt.clf()
plt.plot(t* 100000, exp_function(t, *params2_raw) , '-b', label='Fit-Funktion')
plt.plot(time*100000, amplitude, 'rx', label = 'Messwerte')
plt.xlim(-1, 37)
plt.ylim()
plt.grid()
plt.legend(loc='best')
plt.xlabel('Zeit t in $10^{-5}$ s')
plt.ylabel('$A(t)$ / V')
plt.savefig('amplitude.pdf')
