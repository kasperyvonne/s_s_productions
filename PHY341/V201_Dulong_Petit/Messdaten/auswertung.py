import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from pint import UnitRegistry
import scipy.constants as const
import matplotlib.pyplot as plt
u = UnitRegistry()
Q_ = u.Quantity
R = const.gas_constant
c_wasser = Q_(4.18, 'joule/(gram kelvin)')
m_wasser = Q_(1, 'kilogram')
m_wasser_x = Q_(1, 'gram')
m_wasser_y = Q_(1, 'gram')
T_y = Q_(22, 'degC')
T_x = Q_(18, 'degC')
T_misch_for_cgmg = Q_(20, 'degC')
print(3 * R)
#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin, ampere, kilogram, gram, pascal, bar, speed_of_light, mol


#einlesen der konstanten
rho_graphit_raw, M_graphit_raw, alpha_graphit_raw, kappa_graphit_raw = np.genfromtxt('materialkonstanten_graphit.txt', unpack=True)
rho_graphit = Q_(rho_graphit_raw, 'gram/centimeter^3')
M_graphit = Q_(M_graphit_raw, 'gram/mol')
alpha_graphit = Q_(alpha_graphit_raw * 1e-06, '1/kelvin')
kappa_graphit = Q_(kappa_graphit_raw * 1e09, 'kilogram/(second^2 meter)')
mol_vol_graphit = M_graphit / rho_graphit
print(mol_vol_graphit)


print(rho_graphit.to('kilogram/m^3'))

#U_th in mV
U_th = np.array(np.ones(10))
print(U_th)

def temp (U):
    return Q_(25.157*U - 0.19* U**2, 'celsius')

def CptoCv(cp, alpha, kappa, molvol, T):
    return cp - 9 * alpha**2 * kappa * molvol * T

print(temp(U_th))

c_g_m_g = (c_wasser * m_wasser_y * (T_y - T_misch_for_cgmg)  -  c_wasser * m_wasser_x*(T_misch_for_cgmg - T_x))/(T_misch_for_cgmg - T_x)

print(c_g_m_g)
