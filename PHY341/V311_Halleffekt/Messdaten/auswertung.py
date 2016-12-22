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
#Abmessungen der Proben
d_zink = Q_(1.85-1.70, 'millimeter')
d_kupfer = Q_(18 * 1e-06, 'millimeter')
print(d_zink, d_kupfer)

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:

def E_fermi(n):
	return (h**2 /(2 * m_0) * ( ( (3 * n) / (8 * np.pi) ) **2 ) **(1/3) ).to('eV')
#n_test = Q_(3, '1/(m**3)')
#print(E_fermi(n_test).to('joule'))
def F_1(x, m, b):
	return m * x + b

I_eich_steigend, B_eich_steigend = np.genfromtxt('flussdichte_steigend.txt', unpack=True)
I_eich_fallend, B_eich_fallend = np.genfromtxt('flussdichte_fallend.txt', unpack=True)
params_1, covariance_1 = curve_fit(F_1, I_eich_steigend, B_eich_steigend, sigma=0.1)
errors_B = np.sqrt(np.diag(covariance_1))
params_B = unp.uarray(params_1, errors_B)
print('Gerade B(I) Paramter: ', params_B)
I_lim = np.linspace(0, 10, 100)
plt.plot(I_eich_steigend, B_eich_steigend, 'bx', label='Messwerte steigender Strom')
plt.plot(I_eich_fallend, B_eich_fallend, 'rx', label='Messwerte fallender Strom')
plt.plot(I_lim, F_1(I_lim, *params_1), '-b', label='Lineare Regression')
plt.xlim(I_eich_steigend[0], I_eich_steigend[-1])
plt.xlabel('$I$ in $A$')
plt.ylabel('$B$ in $mT$')
plt.ylim(-1500, 500)
plt.grid()
plt.legend()
plt.savefig('hysterese.pdf')

with open('hysterese_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write(3 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I_q$ in $\si{\\ampere}$} & {$B_{wachsend}$ in $\si{\milli \\tesla}$}  &  {$B_{fallend}$ in $\si{\milli \\tesla}$}  \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_eich_steigend)):
        f.write('{:.1f} & {:.1f} & {:.1f}  \\\ \n'.format(I_eich_steigend[i], B_eich_steigend[i], B_eich_fallend[-(i+1)]))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Messung des magnetischen Feldes bei fallendem und steigendem Strom} \n \\label{tab: hysterese} \n  \\end{table}')

def B(I):
	return Q_(params_1[0] * I + params_1[1], 'millitesla')

#Bestimmung Widerstand
I_zink_raw, U_zink_raw = np.genfromtxt('uri_zink.txt', unpack=True)
I_zink = Q_(I_zink_raw, 'ampere')
U_zink = Q_(U_zink_raw, 'millivolt')
params_zink_R, cov_zink_R = curve_fit(F_1, I_zink.magnitude, U_zink.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(0, 10, 100)
plt.plot(I_zink.magnitude, U_zink.magnitude, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_zink_R), '-b', label='Lineare Regression')
plt.xlim(0, 8)
plt.ylim(0, 80)
plt.xlabel('$I$ in $A$')
plt.ylabel('$U$ in $mV$')
plt.grid()
plt.legend()
plt.savefig('uri_zink.pdf')

R_zink_errors = np.sqrt(np.diag(cov_zink_R))
R_zink = Q_(ufloat(params_zink_R[0], R_zink_errors[0]), 'millivolt/ampere').to('milliohm')
print('R_zink = ', R_zink)

with open('uri_zink_tab.tex', 'w') as f:
    f.write(' \\begin{tabular}{')
    f.write(2 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$U$ in $\si{\\volt}$}   \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_zink_raw)):
        f.write('{:.1f} & {:.1f}  \\\ \n'.format(I_zink_raw[i], U_zink_raw[i]))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Zinkprobe: Messung der Spannung in Abhängigkeit vom Strom } \n \\label{tab: uri_zink}')


I_kupfer_raw, U_kupfer_raw = np.genfromtxt('uri_kupfer.txt', unpack=True)
I_kupfer = Q_(I_kupfer_raw, 'ampere')
U_kupfer = Q_(U_kupfer_raw, 'millivolt')
params_kupfer_R, cov_kupfer_R = curve_fit(F_1, I_kupfer.magnitude, U_kupfer.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(0, 10, 100)
plt.plot(I_kupfer.magnitude, U_kupfer.magnitude, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_kupfer_R), '-b', label='Lineare Regression')
plt.xlim(0, 8)
plt.xlabel('$I$ in $A$')
plt.ylabel('$U$ in $mV$')
plt.grid()
plt.legend()
plt.savefig('uri_kupfer.pdf')
R_kupfer_errors = np.sqrt(np.diag(cov_kupfer_R))
R_kupfer = Q_(ufloat(params_kupfer_R[0], R_kupfer_errors[0]), 'millivolt/ampere').to('milliohm')
print('R_kupfer = ', R_kupfer)

with open('uri_kupfer_tab.tex', 'w') as f:
    f.write(' \\begin{tabular}{')
    f.write(2 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$U$ in $\si{\\volt}$}   \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_kupfer_raw)):
        f.write('{:.1f} & {:.1f}  \\\ \n'.format(I_kupfer_raw[i], U_kupfer_raw[i]))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Kupferprobe: Messung der Spannung in Abhängigkeit vom Strom} \n \\label{tab: uri_kupfer}')





#Hallspannung Kupfer
#konst. B_Feld bei I_q = 3A
B_konst_kupfer = B(3)
U_ges_min_kupfer_konstB, U_ges_plu_kupfer_konstB = np.genfromtxt('u_h_konstB_kupfer.txt', unpack=True)
U_h_kupfer_konstB = Q_(0.5 * (U_ges_plu_kupfer_konstB - U_ges_min_kupfer_konstB), 'millivolt')
I = np.linspace(0, 10 , 11)
with open('u_h_kupfer_konstB_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write(4 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I)):
        f.write('{:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I[i], U_ges_min_kupfer_konstB[i], U_ges_plu_kupfer_konstB[i], U_h_kupfer_konstB[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Hallspannung Kupfer bei konstantem Magnetfeld} \n \\label{tab: hall_kupfer_konstB} \n  \\end{table}')

params_kupfer_U_h_1, cov_kupfer_U_h_1 = curve_fit(F_1, I, U_h_kupfer_konstB.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(0, 10, 100)
plt.plot(I, U_h_kupfer_konstB, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_kupfer_U_h_1), '-b', label='Lineare Regression')
plt.xlim(0, 8)
plt.xlabel('$I$ in $A$')
plt.ylabel('$U_H$ in $mV$')
plt.grid()
plt.legend()
plt.savefig('u_h_kupfer_konstB.pdf')

Steigung_U_h_kupfer_konstB_errors = np.sqrt(np.diag(cov_kupfer_U_h_1))
Steigung_U_h_kupfer_konstB = Q_(ufloat(params_kupfer_U_h_1[0], Steigung_U_h_kupfer_konstB_errors[0]), 'meter^3 * millitesla /(coulomb * millimeter)')
print('Steigung der Hall Spannung, Kupfer, konst BFeld: ', Steigung_U_h_kupfer_konstB.to('millivolt/ampere'))
n_kupfer_konstB =  - 1/(Steigung_U_h_kupfer_konstB * e_0 * d_kupfer) * B_konst_kupfer
print('n_kupfer_konstB: ', n_kupfer_konstB)
print('Fermienergie Kupfer: ', E_fermi(n_kupfer_konstB))

B_konst_zink = B(3)
U_ges_min_zink_konstB, U_ges_plu_zink_konstB = np.genfromtxt('u_h_konstB_zink.txt', unpack=True)
U_h_zink_konstB = Q_(0.5 * (U_ges_plu_zink_konstB - U_ges_min_zink_konstB), 'millivolt')
I = np.linspace(0, 10 , 11)
with open('u_h_zink_konstB_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write(4 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I)):
        f.write('{:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I[i], U_ges_min_zink_konstB[i], U_ges_plu_zink_konstB[i], U_h_zink_konstB[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Hallspannung Zink bei konstantem Magnetfeld} \n \\label{tab: hall_zink_konstB} \n  \\end{table}')

params_zink_U_h_1, cov_zink_U_h_1 = curve_fit(F_1, I, U_h_zink_konstB.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(0, 10, 100)
plt.plot(I, U_h_zink_konstB, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_zink_U_h_1), '-b', label='Lineare Regression')
plt.xlim(0, 8)
plt.xlabel('$I$ in $A$')
plt.ylabel('$U_H$ in $mV$')
plt.grid()
plt.ylim(-0.015, 0.005)
plt.legend()
plt.savefig('u_h_zink_konstB.pdf')

Steigung_U_h_zink_konstB_errors = np.sqrt(np.diag(cov_zink_U_h_1))
Steigung_U_h_zink_konstB = Q_(ufloat(params_zink_U_h_1[0], Steigung_U_h_zink_konstB_errors[0]), 'meter^3 * millitesla /(coulomb * millimeter)')
print('Steigung der Hall Spannung, Zink, konst BFeld: ', Steigung_U_h_zink_konstB.to('millivolt/ampere'))
n_zink_konstB =   1/(Steigung_U_h_zink_konstB * e_0 * d_zink) * B_konst_zink
print('n_zink_konstB: ', n_zink_konstB)
print('Fermienergie Zink: ', E_fermi(n_zink_konstB))



#Berechnung UH variables B Feld
konstI = Q_(10, 'ampere')
I_konstI, U_ges_plu_zink_konstI, U_ges_min_zink_konstI = np.genfromtxt('u_h_konstI_zink.txt', unpack=True)
U_h_zink_konstI = Q_(0.5 * (U_ges_plu_zink_konstI - U_ges_min_zink_konstI), 'millivolt')
with open('u_h_zink_konstI_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write(5 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$B$ in $\si{\milli\\tesla}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_konstI)):
        f.write('{:.1f} & {:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I_konstI[i],B(I_konstI[i]).magnitude ,U_ges_min_zink_konstI[i], U_ges_plu_zink_konstI[i], U_h_zink_konstI[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Hallspannung Zink bei konstantem Querstrom} \n \\label{tab: hall_zink_konstI} \n  \\end{table}')

B_konstI_zink = B(I_konstI).magnitude
params_zink_U_h_2, cov_zink_U_h_2 = curve_fit(F_1, B_konstI_zink[:-1], U_h_zink_konstI.magnitude[:-1], sigma=0.1)
plt.clf()
B_lim = np.linspace(B_konstI_zink[0], B_konstI_zink[-2], 100)
plt.plot(B_konstI_zink[:-1], U_h_zink_konstI.magnitude[:-1], 'rx', label='Messwerte')
plt.plot(B_lim, F_1(B_lim, *params_zink_U_h_2), '-b', label='Lineare Regression')
plt.xlabel('$B$ in $mT$')
plt.ylabel('$U_H$ in $mV$')
plt.xlim(B_konstI_zink[-2], B_konstI_zink[0])
plt.grid()
plt.legend(loc ='best')
plt.savefig('u_h_zink_konstI.pdf')

Steigung_U_h_zink_konstI_errors = np.sqrt(np.diag(cov_zink_U_h_2))
Steigung_U_h_zink_konstI = Q_(ufloat(params_zink_U_h_2[0], Steigung_U_h_zink_konstI_errors[0]), 'meter^3 * ampere /(coulomb * millimeter)')
print('Steigung der Hall Spannung, Zink, konst Strom: ', Steigung_U_h_zink_konstI.to('volt/tesla'))
n_zink_konstI =   1/(Steigung_U_h_zink_konstI * e_0 * d_zink) * konstI
print('n_zink_konstI: ', n_zink_konstI)





konstI = Q_(10, 'ampere')
I_konstI, U_ges_plu_kupfer_konstI, U_ges_min_kupfer_konstI = np.genfromtxt('u_h_konstI_kupfer.txt', unpack=True)
U_h_kupfer_konstI = Q_(0.5 * (U_ges_plu_kupfer_konstI - U_ges_min_kupfer_konstI), 'millivolt')
with open('u_h_kupfer_konstI_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write(5 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$B$ in $\si{\milli\\tesla}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_konstI)):
        f.write('{:.1f} & {:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I_konstI[i] ,B(I_konstI[i]).magnitude ,U_ges_min_kupfer_konstI[i], U_ges_plu_kupfer_konstI[i], U_h_kupfer_konstI[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Hallspannung Kupfer bei konstantem Querstrom} \n \\label{tab: hall_kupfer_konstI} \n  \\end{table}')

B_konstI_kupfer = B(I_konstI).magnitude
params_kupfer_U_h_2, cov_kupfer_U_h_2 = curve_fit(F_1, B_konstI_kupfer, U_h_kupfer_konstI.magnitude, sigma=0.1)
plt.clf()
B_lim = np.linspace(B_konstI_kupfer[0], B_konstI_kupfer[-1], 100)
plt.plot(B_konstI_kupfer[:-1], U_h_kupfer_konstI.magnitude[:-1], 'rx', label='Messwerte')
plt.plot(B_lim, F_1(B_lim, *params_kupfer_U_h_2), '-b', label='Lineare Regression')
plt.xlabel('$B$ in $mT$')
plt.ylabel('$U_H$ in $mV$')
plt.xlim(B_konstI_kupfer[-1], B_konstI_kupfer[0])
plt.grid()
plt.legend(loc ='best')
plt.savefig('u_h_kupfer_konstI.pdf')

Steigung_U_h_kupfer_konstI_errors = np.sqrt(np.diag(cov_kupfer_U_h_2))
Steigung_U_h_kupfer_konstI = Q_(ufloat(params_kupfer_U_h_2[0], Steigung_U_h_kupfer_konstI_errors[0]), 'meter^3 * ampere /(coulomb * millimeter)')
print('Steigung der Hall Spannung, Kupfer, konst Strom: ', Steigung_U_h_kupfer_konstI.to('volt/tesla'))
n_kupfer_konstI =  - 1/(Steigung_U_h_kupfer_konstI * e_0 * d_kupfer) * konstI
print('n_kupfer_konstI: ', n_kupfer_konstI)
