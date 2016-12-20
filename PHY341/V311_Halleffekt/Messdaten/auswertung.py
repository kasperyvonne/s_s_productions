import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import scipy.constants as const
u = UnitRegistry()
Q_ = u.Quantity
h = Q_(const.h, 'joule*second')
e_0 = Q_(const.elementary_charge, 'coulomb')
m_0 = Q_(const.m_e, 'kilogram')
print (h, m_0, e_0)
#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:

def E_fermi(n):
	return h**2 /(2 * m_0) * ( ( (3 * n) / (8 * np.pi) ) **2 ) **(1/3)
#n_test = Q_(3, '1/(m**3)')
#print(E_fermi(n_test).to('joule'))
def F_1(x, m, b):
	return m * x + b

I_eich_steigend, B_eich_steigend = np.genfromtxt('flussdichte_steigend.txt', unpack=True)
I_eich_fallend, B_eich_fallend = np.genfromtxt('flussdichte_fallend.txt', unpack=True)
params_1, covariance_1 = curve_fit(F_1, I_eich_steigend, B_eich_steigend, sigma=0.1)
I_lim = np.linspace(0, 10, 100)
#plt.plot(I_eich_steigend, -B_eich_steigend, 'rx')
#plt.plot(I_eich_fallend, -B_eich_fallend, 'bx')
#plt.plot(I_lim, -F_1(I_lim, *params_1), '-r')
#plt.show()

def B(I):
	return params_1[0] * I + params_1[1]






#Plotbereich

#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()
#
#plt.plot(,,'rx',label='')
#
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('.pdf')
