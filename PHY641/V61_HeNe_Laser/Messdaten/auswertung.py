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
from scipy.stats import sem
import latex as l
#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS
a = [1.1, 2.2, 3.3]
b = [0.1, 0.2, 0.3]
c = [3, 4, 5]
d = [6, 7, 8] #EXAMPLE ARRAYS

f = unp.uarray(a, b) #EXAMPLE UARRAY

l.Latexdocument('latex_example.tex').tabular(
data = [c, d, f], #Data incl. unpuarray
header = ['\Delta Q / \giga\elementarycharge', 'T_1 / \micro\second', 'T_1 / \micro\second'],
places = [1, 1, (1.1, 1.1)],
caption = 'testcaption.',
label = 'testlabel')
##############################################

#r.makeresults()

def E_00(d,I_0,d_0, omega):
    return I_0*np.exp( -2*( (d-d_0)/omega )**2 )

def E_10(d, I_01, d_01, omega_1, I_02, d_02, omega_2):
    return I_01 * np.exp( -2 * ( (d-d_01) / omega_1 )**2 )+ I_02 * np.exp( -2 * ( ( d-d_02) / omega_2 )**2 )

def polarisation( phi, I_0, phi_0):
    return I_0*( np.sin(phi-phi_0) )**2

def lamb(spaltbreite, n, abstand_haupt, abstand_neben):
    bib={}
    bib['theta']=unp.arctan(abstand_neben/abstand_haupt)
    bib['lambda']=spaltbreite/n*unp.sin(bib['theta'])
    return bib






###################
# Auswertung T_00 #
###################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Grundmode T_00', '\n', '-----------------------------------------------------------\n\n\n')
r_00, I_00=np.genfromtxt('T_00.txt',unpack=True)
params_I_00, cov_I_00 = curve_fit(E_00,r_00,I_00 )
error_I_00= np.sqrt(np.diag(cov_I_00))

amplitude_I_00 = ufloat(params_I_00[0],error_I_00[0])
verschiebung_I_00 = ufloat( params_I_00[1], error_I_00[1])
omega_I_00 = ufloat( params_I_00[2], error_I_00[2])

print('Amplitude: ', amplitude_I_00,'\n')
print('Verschiebung ', verschiebung_I_00,'\n')
print('Frequenz ', omega_I_00, '\n')

# Tabelle erstellen

l.Latexdocument('./table/T_00.tex').tabular(
data = [r_00, I_00], #Data incl. unpuarray
header = ['r / \milli\meter ', 'I_p / \micro\ampere'],
places = [1, 1],
caption = 'Messwerte der T_00 Mode.',
label = 'T_00')


# Plotten
d_00=np.linspace(r_00[0]-1,r_00[-1]+1,10000)

plt.plot(r_00, I_00, '.', label='Messwerte')
plt.plot(d_00, E_00(d_00, params_I_00[0], params_I_00[1], params_I_00[2]), label='Fit')
plt.legend()
plt.xlabel(r'$r \, / \, mm $')
plt.ylabel(r'$ I_{\mathrm{p}}\, / \, \mu A$')
plt.grid()
plt.savefig('./plots/T_00.pdf')
#plt.show()




###################
# Auswertung T_10 #
###################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Grundmode T_10', '\n', '-----------------------------------------------------------\n\n\n')

r_10, I_10=np.genfromtxt('T_10_2.txt',unpack=True)


I_max_left=max(I_10[1:10])
I_max_right=max(I_10[10:-1])
null=np.where(I_10<0.04)[0][0]
print(null)
r_left_max=r_10[np.where(I_10==I_max_left)[0][0]]
r_right_max=r_10[np.where(I_10==I_max_right)[0][0]]

omega_guess_left=8*sem( I_10[1:13])
omega_guess_right=8*sem( I_10[13:-1])

print([I_max_left, r_left_max, omega_guess_left, I_max_right, r_right_max, omega_guess_right])

params_I_10, cov_I_10 = curve_fit(E_10,r_10,I_10,p0=[I_max_left, r_left_max, 1, I_max_right, r_right_max,1])
error_I_10= np.sqrt(np.diag(cov_I_10))

amplitude_I_01 = ufloat(params_I_10[0],error_I_10[0])
verschiebung_d_01 = ufloat( params_I_10[1], error_I_10[1])
omega_1 = ufloat( params_I_10[2], error_I_10[2])

amplitude_I_02 = ufloat(params_I_10[3],error_I_10[3])
verschiebung_d_02 = ufloat( params_I_10[4], error_I_10[4])
omega_2 = ufloat( params_I_10[5], error_I_10[5])


print('Amplitude 1: ', amplitude_I_01,'\n')
print('Verschiebung 1: ', verschiebung_d_01,'\n')
print('Frequenz 1: ', omega_1, '\n')

print('Amplitude 2: ', amplitude_I_02,'\n')
print('Verschiebung 2: ', verschiebung_d_02,'\n')
print('Frequenz 2: ', omega_2, '\n')

# Tabelle erstellen
l.Latexdocument('./table/T_10.tex').tabular(
data = [r_10, I_10], #Data incl. unpuarray
header = ['r / \milli\meter ', 'I_p / \micro\ampere'],
places = [1, 1],
caption = 'Messwerte der T_10 Mode.',
label = 'T_10')


# Plotten
d_10=np.linspace(r_10[0]-1,r_10[-1]+1,10000)
plt.clf()
plt.plot(r_10, I_10, '.', label='Messwerte')
plt.plot(d_10, E_10( d_10,params_I_10[0], params_I_10[1], params_I_10[2], params_I_10[3], params_I_10[4], params_I_10[5]), label='Fit')
plt.legend()
plt.xlabel(r'$r \, / \, mm $')
plt.ylabel(r'$ I_{\mathrm{p}}\, / \, \mu A$')
plt.grid()
plt.show()





###########################
# Auswertung Polarisation #
###########################
print('\n\n\n-----------------------------------------------------------\n','Auswertung der Polarisation', '\n', '-----------------------------------------------------------\n\n\n')

winkel, I_pola= np.genfromtxt('polarisation.txt', unpack=True)
winkel=np.deg2rad(winkel)
print(max(I_pola))

params_pola, cov_pola = curve_fit(polarisation,winkel,I_pola)
error_pola= np.sqrt(np.diag(cov_pola))

amplitude_pola=ufloat( params_pola[0], error_pola[0])
phase_pola= ufloat( params_pola[1], error_pola[1])

print(' Amplitude Pola: ', amplitude_pola,'\n')
print(' Phase: ', phase_pola, '\n')

# Erstellen der Tabelle

l.Latexdocument('./table/polar.tex').tabular(
data = [winkel, I_pola], #Data incl. unpuarray
header = ['\phi / \rad ', 'I_p / \milli\ampere'],
places = [1, 1],
caption = 'Aufgenommene Werte bei der Polarisationsmessungs.',
label = 'pola')

# Plot

phi= np.linspace(winkel[0], winkel[-1],1000)
plt.clf()
plt.plot( winkel, I_pola, '.', label='Messdaten')
plt.plot( phi, polarisation(phi, params_pola[0], params_pola[1]), label='Fit')
plt.xlabel(r'$\phi \, / \, \mathrm{rad}$')
plt.ylabel( r'$ I_{\mathrm{p}} \, / \, mA$')
plt.legend()
plt.grid()
plt.xticks([0,0.25*np.pi,0.5*np.pi,0.75*np.pi,np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi, 2*np.pi],['0','$\\frac{1}{4}\,\\pi$', '$\\frac{1}{2}\,\\pi$','$\\frac{3}{4}\,\\pi$' ,'$\\pi$','$\\frac{5}{4}\,\\pi$','$\\frac{3}{2}\, \\pi$','$\\frac{7}{4}\, \\pi$', '$2\, \\pi$'])
plt.savefig('./plots/pola.pdf')






## Auswertung Wellenlänge
print('\n\n\n-----------------------------------------------------------\n','Wellenlängenbestimmung', '\n', '-----------------------------------------------------------\n\n\n')

n, abstand= np.genfromtxt('wellenlaenge.txt', unpack=True)
wellenlange=lamb( 1e-5, n, 83.0e-2,(abstand/2)*1e-2) # dictenoary with 'theta' and 'lambda'

# Tabelle erstellen
l.Latexdocument('wellenlaenge.tex').tabular(
data = [n, abstand, wellenlange['theta'], wellenlange['lambda']], #Data incl. unpuarray
header = ['Ordnung ', 'd / \centi\meter', '\theta / \rad', '\lambda / \nano\meter'],
places = [1, 1, 1, 1],
caption = 'Aufgenommene Messwerte für die Wellenlängenbestimmung. Der Winkel $\theta$ und die Wellenlänge $\lambda$ werden mit den Gleichung \eqref{} und \eqref{} bestimmt. Der Abstand zum Schirm beträgt $l=\SI{80}{\centi\meter}$ und der Gitterabstand $a=\SI{}{}$.',
label = 'wellenlänge')

mittelwert_wellenlaenge= ufloat( np.mean(wellenlange['lambda']),sem(wellenlange['lambda']) )
print('Gemittelte Wellenlange: ', mittelwert_wellenlaenge,'\n')
