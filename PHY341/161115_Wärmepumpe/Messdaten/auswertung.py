import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
std_devs as stds)

#Temperaturen einlesen
T_error = 1
T_1_raw = np.genfromtxt('T_1.txt', unpack = True)
T_2_raw = np.genfromtxt('T_2.txt', unpack = True)
T_1 = unp.uarray(T_1_raw, len(T_1) * [T_error])
T_2 = unp.uarray(T_2_raw, len(T_2) * [T_error])
timeinterval = 60
t = np.linspace(0, (len(T_1) - 1) * timeinterval, num = len(T_1))
print(t)
#Drücke einlesen
p_error = 1
p_a_raw = np.genfromtxt('p_a.txt', unpack = True)
p_b_raw = np.genfromtxt('p_b.txt', unpack = True)
#Konstanten
rho_H2O = 1
vol_1 = 1
vol_2 = 1
kappa = 1.14
# in g/dm^3, bei T = 0 und p = 1 bar
rho_0_CI2F2C = 5.51



#Massen
m_1 =
m_2 = 1
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
paramsF1_T1, covarianceF1_T1 = curve_fit(F1, t, T_1, sigma=0.1)
errorsF1_T1 = np.sqrt(np.diag(covarianceF1_T1))
A_F1_T1 = ufloat(paramsF1_T1[0], errorsF1_T1[0])
B_F1_T1 = ufloat(paramsF1_T1[1], errorsF1_T1[1])
C_F1_T1 = ufloat(paramsF1_T1[2], errorsF1_T1[2])

paramsF1_T2, covarianceF1_T2 = curve_fit(F1, t, T_2, sigma=0.1)
errorsF1_T2 = np.sqrt(np.diag(covarianceF1_T2))
A_F1_T2 = ufloat(paramsF1_T2[0], errorsF1_T2[0])
B_F1_T2 = ufloat(paramsF1_T2[1], errorsF1_T2[1])
C_F1_T2 = ufloat(paramsF1_T2[2], errorsF1_T2[2])


paramsF2, covarianceF2 = curve_fit(F2, t, T_1,  sigma=0.1)
errorsF2 = np.sqrt(np.diag(covarianceF2))

paramsF3, covarianceF3 = curve_fit(F3, t, T_1,  sigma=0.1)
errorsF3 = np.sqrt(np.diag(covarianceF3))

plt.plot(t, T_1, 'rx', label="gemessene Temperaturen T_1")
plt.plot(t, F1(t, *paramsF1_T1), 'b-', label='Fit, F1')
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

#hier später Berechnung von L
L = 1

#Massendurchsatz
dm_dt = dQ2_dt / L


#mechanische Arbeit
#ideale Gasgleichung pV = nRT equals p = nRT/V  
def rho(T):

    return n * R * T / V

N_mech = unp.uarray(np.zeros(4), np.zeros(4))
for i in range(0,4):
    N_mech[i] =  1/(kappa-1) * (p_b[i+1] * (p_a[i]/p_b[i+1])**(1/kappa) - p_a[i+1] ) * 1/rho_0_CI2F2C * dm_dt[i]
