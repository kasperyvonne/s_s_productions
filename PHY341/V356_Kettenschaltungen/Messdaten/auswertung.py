import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
import latex
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
u = UnitRegistry()
Q_ = u.Quantity

#Apparaturkonstanten
L = Q_(1, 'henry')
C = Q_(1, 'farad')
C_1 = Q_(1, 'farad')
C_2 = Q_(2, 'farad')
theta = np.linspace(0, 0.5 * np.pi, 1000)

#Dispersionskurven
#gleiche kondensatoren
def omega(theta):
    return np.sqrt( 2 / (L.magnitude * C.magnitude) * (1 - np.cos(theta) ) )

def nu(omega):
    return 1/(2*np.pi) * omega

def omega1(theta):
    return np.sqrt( 1/ L.magnitude * (1/C_1.magnitude + 1/C_2.magnitude) + 1/L.magnitude*np.sqrt( (1/C_1.magnitude + 1/C_2.magnitude)**2 - 4*np.sin(theta)**2/(C_1.magnitude*C_2.magnitude) ))

def omega2(theta):
    return np.sqrt( 1/ L.magnitude * (1/C_1.magnitude + 1/C_2.magnitude) - 1/L.magnitude*np.sqrt( (1/C_1.magnitude + 1/C_2.magnitude)**2 - 4*np.sin(theta)**2/(C_1.magnitude*C_2.magnitude) ))

def v_phase(nu):
    return 2*np.pi * nu / ( np.arccos(1 - 0.5 * (2 * np.pi * nu)**2 * L.magnitude * C.magnitude) )

def v_gruppe(nu):
    return np.sqrt( 1/(L.magnitude * C.magnitude) * (1 - 0.25 * L.magnitude * C.magnitude * (2*np.pi*nu))    )
#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)


nu = nu(omega(theta))




#Theorieplots der Disperionsrelation
plt.plot(theta, omega(theta), label='Dispersion' )
plt.ylabel('Kreisfrequenz $\\omega$ in $1/s$')
plt.xlabel('Phasenverschiebung $\\theta$')
plt.legend(loc='best')
plt.grid()
plt.savefig('dispersion.pdf')

plt.clf()
plt.plot(theta, omega1(theta), label='$\omega_1(\\theta)$' )
plt.plot(theta, omega2(theta), label='$\omega_2(\\theta)$' )
plt.ylabel('Kreisfrequenz $\\omega$ in $1/s$')
plt.xlabel('Phasenverschiebung $\\theta$')
plt.legend(loc='best')
plt.grid()
plt.savefig('dispersion1.pdf')


plt.clf()
plt.plot( nu, v_phase(nu), label='$v_{Ph}(\\nu)$' )
plt.ylabel('Phasengeschwindigkeit $v$ in $m/s$')
plt.xlabel('Frequenz $\\nu$ in $1/s$')
plt.legend(loc='best')
plt.grid()
plt.savefig('v_phase.pdf')

plt.clf()
plt.plot( nu, v_gruppe(nu), label='$v_{Gr}(\\nu)$' )
plt.ylabel('Gruppengeschwindigkeit $v$ in $m/s$')
plt.xlabel('Frequenz $\\nu$ in $1/s$')
plt.legend(loc='best')
plt.grid()
plt.savefig('v_gruppe.pdf')
