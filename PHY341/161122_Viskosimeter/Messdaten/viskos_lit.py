import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms,
std_devs as stds)
import scipy.constants as const

T, n = np.genfromtxt('viskositaet.txt', unpack = True)
T_kohl, n_kohl = np.genfromtxt('viskositaet_kohlrausch.txt', unpack = True)
n *= 1000
n_kohl 

def F1(t, A, B):
    return A * np.exp(B / t)


paramsF1, covarianceF1 = curve_fit(F1, T, n)
errorsF1 = np.sqrt(np.diag(covarianceF1))

x_t = np.linspace(T[0], T[-1] , num = 1000)
plt.plot(T, n, 'rx', label="Literaturwerte $\eta$")
plt.plot(x_t, F1(x_t, *paramsF1), 'b-', label='Regressionskurve')
plt.xlim(T[0], T[-1])
plt.xlabel('Temperatur in $Celsius$')
plt.grid(which="both")
plt.ylabel('Viskosität in $Pa s$')
plt.legend(loc="best")
plt.savefig('literaturwerte.pdf')


plt.clf()
plt.xlim(1/T[0], 1/T[-1])
plt.plot(1/T[0:-1], n[0:-1],'rx', label="Literatur $\eta$")
plt.plot(1/x_t, F1(x_t, *paramsF1), 'b-', label = "Regressionsgerade")
plt.yscale("log")
plt.ylabel('$\eta$')
plt.xlabel('$1 durch Temperatur $')
plt.grid(which="both")
plt.legend(loc="best")
plt.savefig('halblog.pdf')



paramsF1_kohl, covarianceF1_kohl = curve_fit(F1, T_kohl, n_kohl)
errorsF1_kohl = np.sqrt(np.diag(covarianceF1_kohl))
print(paramsF1_kohl)
plt.clf()
x_t = np.linspace(T_kohl[0], T_kohl[-1] , num = 1000)
plt.plot(T_kohl, n_kohl, 'rx', label="Literaturwerte $\eta$")
plt.plot(x_t, F1(x_t, *paramsF1_kohl), 'b-', label='Regressionskurve')
plt.xlim(T_kohl[0], T_kohl[-1])
plt.xlabel('Temperatur in $Celsius$')
plt.grid(which="both")
plt.ylabel('Viskosität in $Pa s$')
plt.legend(loc="best")
plt.savefig('literaturwerte_kohl.pdf')


plt.clf()
plt.xlim(1/10, 1/100)
plt.plot(1/T_kohl, n_kohl,'rx', label="Literatur $\eta$")
plt.plot(1/x_t, F1(x_t, *paramsF1_kohl), 'b-', label = "Regressionsgerade")
plt.yscale("log")
plt.ylabel('$\eta$')
plt.xlabel('$1 durch Temperatur $')
plt.grid(which="both")
plt.legend(loc="best")
plt.savefig('halblog_kohl.pdf')
