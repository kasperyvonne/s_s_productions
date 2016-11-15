import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
std_devs as stds)
import scipy.constants as const
#Temperaturen einlesen
T_error = 1
T_1_raw = np.genfromtxt('T_1.txt', unpack = True)
T_2_raw = np.genfromtxt('T_2.txt', unpack = True)
T_1 = unp.uarray(T_1_raw, len(T_1_raw) * [T_error])
T_2 = unp.uarray(T_2_raw, len(T_2_raw) * [T_error])
#umrechnung kalvin
T_1 += 273.15
T_2 += 273.15
timeinterval = 60
t = np.linspace(0, (len(T_1) - 1) * timeinterval, num = len(T_1))
print(t)
#Drücke einlesen
p_error = 1
p_a_raw = np.genfromtxt('p_a.txt', unpack = True)
p_b_raw = np.genfromtxt('p_b.txt', unpack = True)
p_a = unp.uarray(p_a_raw, len(p_a_raw) * [p_error])
p_b = unp.uarray(p_a_raw, len(p_b_raw) * [p_error])
#umrechnung pascal
p_a *= 1e05
p_b *= 1e05

#Konstanten
R = const.gas_constant
rho_H2O = 1
vol_1 = 1
vol_2 = 1
kappa = 1.14
# in g/dm^3, bei T = 0 und p = 1 bar
rho_0_CI2F2C = 5.51

#Massen
m_1 = rho_H2O * vol_1
m_2 = rho_H2O * vol_2
m_k = 1
c_w = 1
c_k = 1

#verschieden Funktionen für curve_fit
def F1 (t, A, B, C):
    return A*t**2 + B*t + C

def F2(t, A, B, C):
    return (A)/(1 + B*t**1.5)

def F3(t, A, B, C):
    return (A*t**1.5)/(1 + B*t**1.5) + C

#curve_fit für F1, F2, F3
paramsF1_T1, covarianceF1_T1 = curve_fit(F1, t, noms(T_1), sigma=0.1)
errorsF1_T1 = np.sqrt(np.diag(covarianceF1_T1))
A_F1_T1 = ufloat(paramsF1_T1[0], errorsF1_T1[0])
B_F1_T1 = ufloat(paramsF1_T1[1], errorsF1_T1[1])
C_F1_T1 = ufloat(paramsF1_T1[2], errorsF1_T1[2])

paramsF1_T2, covarianceF1_T2 = curve_fit(F1, t, noms(T_2), sigma=0.1)
errorsF1_T2 = np.sqrt(np.diag(covarianceF1_T2))
A_F1_T2 = ufloat(paramsF1_T2[0], errorsF1_T2[0])
B_F1_T2 = ufloat(paramsF1_T2[1], errorsF1_T2[1])
C_F1_T2 = ufloat(paramsF1_T2[2], errorsF1_T2[2])
print('Gefittete Funktion:', A_F1_T1 , 'x2 +' , B_F1_T1 , 'x + ' , C_F1_T1)

paramsF2, covarianceF2 = curve_fit(F2, t, noms(T_1),  sigma=0.1)
errorsF2 = np.sqrt(np.diag(covarianceF2))

paramsF3, covarianceF3 = curve_fit(F3, t, noms(T_1),  sigma=0.1)
errorsF3 = np.sqrt(np.diag(covarianceF3))

x_t = np.linspace(0, 1000 , num = 1000)
plt.plot(t, noms(T_1), 'rx', label="gemessene Temperaturen T_1")
plt.plot(x_t, F1(x_t, *paramsF1_T1), 'b-', label='Fit, F1')
#plt.plot(t, F2(t, *paramsF2), 'g-', label='Fit, F2')
#plt.plot(t, F3(t, *paramsF3), label='Fit, F3')
plt.legend(loc="best")
plt.savefig('plot.pdf')


#Differentialquotient
def dT_dt(t, A, B):
    return 2*A*t + B
#für 60, 120, 180, 240
dT1_dt = unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    dT1_dt[i] = dT_dt(t[i+1], A_F1_T1, B_F1_T1)

dT2_dt = unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    dT2_dt[i] = dT_dt(t[i+1], A_F1_T2, B_F1_T2)

dQ1_dt =  unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    dQ1_dt[i] = dT1_dt[i] * (m_1 * c_w + m_k * c_k)

dQ2_dt =  unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    dQ2_dt[i] = dT2_dt[i] * (m_2 * c_w + m_k * c_k)

#Berechnung von L
def Gaskurve(T, A, B):
    return A * np.exp(B/T)
L_params, L_cov = curve_fit(Gaskurve, noms(T_1), noms(p_b))
L = - np.sqrt(np.diag(L_cov))[1] * R
print(L)
x_t = np.linspace(0, 1000 , num = 1000)
plt.yscale("log")
plt.errorbar(1/noms(T_1), noms(p_b), yerr=stds(p_b), fmt="rx", label="$p_{b}$")
plt.plot(1/x_t, Gaskurve(noms(T_1), *L_cov), 'rx', label = "fit")
plt.legend(loc="best")
plt.savefig('plot2.pdf')


#Massendurchsatz
dm_dt = dQ2_dt / L


#mechanische Arbeit
#ideale Gasgleichung pV = nRT equals p = nRT/V
const = (T_0 * rho_0_CI2F2C / 1e05)
def rho(p, T):
    return const * (p/T)

N_mech = unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    N_mech[i] =  1/(kappa-1) * (p_b[i+1] * (p_a[i]/p_b[i+1])**(1/kappa) - p_a[i+1] ) * 1/rho(p_a[i+1]) * dm_dt[i]

#Wirkungskoeffizient
nu_real = dQ1_dt / N_mech
print(nu_real)
