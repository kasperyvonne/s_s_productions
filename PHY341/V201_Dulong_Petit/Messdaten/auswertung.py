import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from pint import UnitRegistry
import matplotlib.pyplot as plt
u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin, ampere, kilogram, gram, pascal, bar, speed_of_light, mol
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'cal')
c = Q_(1, 'cal')
s = Q_(1, 'second')
print(c.to('joule'))
print(a**2)
print(b)
###############################

#einlesen der konstanten
rho_graphit_raw, M_graphit_raw, alpha_graphit_raw, kappa_graphit_raw = np.genfromtxt('materialkonstanten_graphit.txt', unpack=True)
rho_graphit = Q_(rho_graphit_raw, 'gram/centimeter^3')
M_graphit = Q_(M_graphit_raw, 'gram/mol')
alpha_graphit = Q_(alpha_graphit_raw * 1e-06, '1/kelvin')
kappa_graphit = Q_(kappa_graphit_raw * 1e09, 'kilogram/(second^2 meter)')
print(rho_graphit, M_graphit, alpha_graphit, kappa_graphit)


print(rho_graphit.to('kilogram/m^3'))

#U_th in mV
U_th = np.array(np.ones(10))
print(U_th)

def temp (U):
    return Q_(25.157*U - 0.19* U**2, 'celsius')

print(temp(U_th))
