import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
import latex
from uncertainties.umath import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
u = UnitRegistry()
Q_ = u.Quantity

#Apparaturkonstanten
L = Q_(1.75e-3, 'henry')
C = Q_(22.0e-9, 'farad')
C_1 = C
C_2 = Q_(9.39e-9, 'farad')
theta = np.linspace(0, np.pi, 100)

#Dispersionskurven
#gleiche kondensatoren
def omega(theta):
    return np.sqrt( 2 / (L.magnitude * C.magnitude) * (1 - np.cos(theta) ) )

def nu(omega):
    return 1/(2*np.pi) * omega


#unterschiedliche Kondensatoren
def omega1(theta):
    return np.sqrt( 1/ L.magnitude * (1/C_1.magnitude + 1/C_2.magnitude) + 1/L.magnitude*np.sqrt( (1/C_1.magnitude + 1/C_2.magnitude)**2 - 4*np.sin(theta)**2/(C_1.magnitude*C_2.magnitude) ))

def omega2(theta):
    return np.sqrt( 1/ L.magnitude * (1/C_1.magnitude + 1/C_2.magnitude) - 1/L.magnitude*np.sqrt( (1/C_1.magnitude + 1/C_2.magnitude)**2 - 4*np.sin(theta)**2/(C_1.magnitude*C_2.magnitude) ))

def f(x, A, B, C):
    return A * np.exp(B * x) + C

#Phasengeschwindigkeit
def v_phase(nu):
    return 2*np.pi * nu / ( np.arccos(1 - 0.5 * (2 * np.pi * nu)**2 * L.magnitude * C.magnitude) )

#Gruppengeschwindigkeit
def v_gruppe(nu):
    return np.sqrt( 1/(L.magnitude * C.magnitude) * (1 - 0.25 * L.magnitude * C.magnitude * (2*np.pi*nu))    )

def impedanz_plot(omega):
    return np.sqrt(L.magnitude / C.magnitude) * 1/np.sqrt( 1 - 0.25 * omega**2 * L.magnitude * C.magnitude )

def impedanz(omega):
    return np.sqrt(L / C) * 1/np.sqrt( 1 - 0.25 * omega**2 * L * C )
#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)


#Einlesen der Messwerte
eigenfrequenzen_a_LC = np.genfromtxt('eigenfrequenzen_a_LC.txt', unpack=True)
range_lin = np.linspace(1, len(eigenfrequenzen_a_LC), 12)
Phasenverschiebung_pro_glied_LC = np.pi * range_lin / 16

eigenfrequenzen_a_LC1C2 = np.genfromtxt('eigenfrequenzen_a_LC1C2.txt', unpack=True)
range_lin = np.linspace(1, len(eigenfrequenzen_a_LC1C2), len(eigenfrequenzen_a_LC1C2))
Phasenverschiebung_pro_glied_LC1C2 = np.pi * range_lin / 16

frequenzen_sweep_LC = np.genfromtxt('frequenz_sweep_LC.txt', unpack = True)
x_range_LC = ((np.linspace(1, len(frequenzen_sweep_LC), len(frequenzen_sweep_LC))-1) * 4)[::-1]
print(x_range_LC)
params_LC, covariance_LC = curve_fit(f, x_range_LC, frequenzen_sweep_LC)
errors_LC = np.sqrt(np.diag(covariance_LC))
A_param = ufloat(params_LC[0], errors_LC[0])
B_param = ufloat(params_LC[1], errors_LC[1])
C_param = ufloat(params_LC[2], errors_LC[2])
def frequenz_sweep_LC(x):
    return A_param * exp(B_param * x) + C_param


frequenzen_sweep_LC1C2 = np.genfromtxt('frequenz_sweep_LC1C2.txt', unpack = True)
x_range_LC1C2 = ((np.linspace(1, len(frequenzen_sweep_LC1C2), len(frequenzen_sweep_LC1C2))-1) * 4)[::-1]
print(x_range_LC1C2)
params_LC1C2, covariance_LC1C2 = curve_fit(f, x_range_LC1C2, frequenzen_sweep_LC1C2)
errors_LC1C2 = np.sqrt(np.diag(covariance_LC1C2))
A_param_LC1C2 = ufloat(params_LC1C2[0], errors_LC1C2[0])
B_param_LC1C2 = ufloat(params_LC1C2[1], errors_LC1C2[1])
C_param_LC1C2 = ufloat(params_LC1C2[2], errors_LC1C2[2])

def frequenz_sweep_LC1C2(x):
    return A_param_LC1C2 * exp(B_param_LC1C2 * x) + C_param_LC1C2

print('Parameter des Fits LC1C2, A= ', A_param_LC1C2, ' B= ', B_param_LC1C2, ' C= ', C_param_LC1C2)
#Berechnungen
#Theoretische Werte
omega_G_LC = np.sqrt(2 / (L * C)).to('1/second')
print('Theoretische Grenzfrequenz LC: ', omega_G_LC)

omega_G_u_LC1C2 = np.sqrt(2 / (L * C_1))
omega_G_o_LC1C2 = np.sqrt(2 / (L * C_2))
print('Theoretische Grenzfrequenz unten LC1C2: ', omega_G_u_LC1C2)
print('Theoretische Grenzfrequenz oben LC1C2: ', omega_G_o_LC1C2)


#Berechnungen
distance_f_g_LC = ufloat(9, 1)
distance_f_g_u_LC1C2 = ufloat(6, 1)
distance_f_g_o_LC1C2 = ufloat(11, 1)
print('Aus Sweep Methode bestimmte Grenzfrequenz LC: ', frequenz_sweep_LC(distance_f_g_LC) * np.pi * 2)
print('Prozentuale Abweichung: ', (frequenz_sweep_LC(distance_f_g_LC) * np.pi * 2)/omega_G_LC.magnitude - 1)
print('Aus Sweep Methode bestimmte Grenzfrequenz unten LC1C2: ', frequenz_sweep_LC1C2(distance_f_g_u_LC1C2) * np.pi * 2)
print('Aus Sweep Methode bestimmte Grenzfrequenz oben LC1C2: ', frequenz_sweep_LC1C2(distance_f_g_o_LC1C2) * np.pi * 2)

nu = nu(omega(theta))
#Theorieplots der Disperionsrelation
plt.plot(theta, omega(theta), label='Dispersionskurve $\omega(\\theta)$' )
plt.plot(Phasenverschiebung_pro_glied_LC, eigenfrequenzen_a_LC * 2 * np.pi, 'rx')
plt.ylabel('Kreisfrequenz $\\omega$ in $1/s$')
plt.xlabel('Phasenverschiebung $\\theta$')
plt.xlim(Phasenverschiebung_pro_glied_LC[0], Phasenverschiebung_pro_glied_LC[-1])
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/dispersion.pdf')


plt.clf()
plt.plot(theta, omega1(theta), label='$\omega_1(\\theta)$' )
plt.plot(theta, omega2(theta), label='$\omega_2(\\theta)$' )
plt.plot(Phasenverschiebung_pro_glied_LC1C2, eigenfrequenzen_a_LC1C2 * 2 * np.pi, 'rx')
plt.ylabel('Kreisfrequenz $\\omega$ in $1/s$')
plt.xlabel('Phasenverschiebung $\\theta$')
plt.xlim(Phasenverschiebung_pro_glied_LC1C2[0], Phasenverschiebung_pro_glied_LC1C2[-1])
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/dispersion1.pdf')


plt.clf()
plt.plot(nu, v_phase(nu), label='$v_{Ph}(\\nu)$' )
plt.ylabel('Phasengeschwindigkeit $v$ in $m/s$')
plt.xlabel('Frequenz $\\nu$ in $1/s$')
plt.xlim(nu[0], nu[-1])
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/v_phase.pdf')


plt.clf()
plt.plot( omega(theta), impedanz_plot(omega(theta)), label='$Z(\omega)$' )
plt.ylabel('Impedanz $Z$')
plt.xlabel('Kreisfrequenz $\omega$ in $1/s$')
plt.xlim(omega(theta)[0], omega(theta)[-1])
plt.legend(loc='best')
plt.grid()
plt.savefig('plots/impedanz.pdf')

x_lim = np.linspace(x_range_LC[0], x_range_LC[-1], 100)
plt.clf()
plt.plot(x_range_LC, frequenzen_sweep_LC, 'rx')
plt.plot(x_lim, f(x_lim, *params_LC), 'b-')
plt.grid()
#plt.plot(x_lim, f(x_lim, *params_LC) )
plt.savefig('plots/frequenzsweep_LC.pdf')

x_lim = np.linspace(x_range_LC1C2[0], x_range_LC1C2[-1], 100)
plt.clf()
plt.plot(x_range_LC1C2, frequenzen_sweep_LC1C2, 'rx')
plt.plot(x_lim, f(x_lim, *params_LC1C2), 'b-')
plt.grid()
plt.savefig('plots/frequenzsweep_LC1C2.pdf')
