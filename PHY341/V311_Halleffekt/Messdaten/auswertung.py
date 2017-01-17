import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
import latex
from pandas import Series, DataFrame
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import scipy.constants as const
u = UnitRegistry()
Q_ = u.Quantity
h = Q_(const.h, 'joule*second')
e_0 = Q_(const.elementary_charge, 'coulomb')
m_0 = Q_(const.m_e, 'kilogram')


#Abmessungen der Proben
d_zink = Q_(1.85 - 1.70, 'millimeter')
d_kupfer = Q_(18e-6, 'meter').to('millimeter')
l_zink = Q_(4.3, 'centimeter')
l_kupfer = Q_(2.805, 'centimeter')
b_kupfer = Q_(2.53, 'centimeter')
b_zink = Q_(2.55, 'centimeter')


def E_fermi(n):
	return ( h**2 /(2 * m_0) * ( ( (3 * n) / (8 * np.pi) ) **2 ) **(1/3) ).to('eV')

#lineare Funktion für Fits
def F_1(x, m, b):
	return m * x + b

I_eich_steigend, B_eich_steigend = np.genfromtxt('flussdichte_steigend.txt', unpack=True)
I_eich_fallend, B_eich_fallend = np.genfromtxt('flussdichte_fallend.txt', unpack=True)
params_1, covariance_1 = curve_fit(F_1, I_eich_steigend, B_eich_steigend, sigma=0.1)
errors_B = np.sqrt(np.diag(covariance_1))
params_B = unp.uarray(params_1, errors_B)

#print('Gerade B(I) Paramter: ', params_B)
I_lim = np.linspace(-0.2, 5.2, 100)
plt.plot(I_eich_steigend, B_eich_steigend, 'bx', label='Messwerte steigender Strom')
plt.plot(I_eich_fallend, B_eich_fallend, 'rx', label='Messwerte fallender Strom')
plt.plot(I_lim, F_1(I_lim, *params_1), '-b', label='Lineare Regression')
plt.xlim(I_eich_steigend[0], I_eich_steigend[-1])
plt.xlabel('$I$ in $A$')
plt.ylabel('$B$ in $mT$')
plt.ylim(-1500, 500)
plt.xlim(I_lim[0], I_lim[-1])
plt.grid()
plt.legend(loc='best')
plt.savefig('hysterese.pdf')

latex.Latexdocument('hysterese_tab.tex').tabular([I_eich_steigend, B_eich_steigend, B_eich_fallend[::-1] ],
'{$I_q$ in $\si{\\ampere}$} & {$B_{wachsend}$ in $\si{\milli \\tesla}$}  &  {$B_{fallend}$ in $\si{\milli \\tesla}$}', [1, 1, 1] ,
caption = 'Messung des magnetischen Feldes bei fallendem und steigendem Strom', label = 'tab: hysterese')


def B(I):
	return Q_(params_1[0] * I + params_1[1], 'millitesla')

#Bestimmung Widerstand
I_zink_raw, U_zink_raw = np.genfromtxt('uri_zink.txt', unpack=True)
I_zink = Q_(I_zink_raw, 'ampere')
U_zink = Q_(U_zink_raw, 'millivolt')
params_zink_R, cov_zink_R = curve_fit(F_1, I_zink.magnitude, U_zink.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(-0.2, 8.2, 100)
plt.plot(I_zink.magnitude, U_zink.magnitude, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_zink_R), '-b', label='Lineare Regression')
plt.xlim(I_zink.magnitude[0], I_zink.magnitude[-1])
plt.ylim(-1, 80)
plt.xlabel('$I$ in $A$')
plt.ylabel('$U$ in $mV$')
plt.xlim(I_lim[0], I_lim[-1])
plt.grid()
plt.legend(loc='best')
plt.savefig('uri_zink.pdf')

R_zink_errors = np.sqrt(np.diag(cov_zink_R))
R_zink = Q_(ufloat(params_zink_R[0], R_zink_errors[0]), 'millivolt/ampere').to('milliohm')
#print('R_zink = ', R_zink)


latex.Latexdocument('uri_zink_tab.tex').tabular([I_zink_raw, U_zink_raw],
'{$I$ in $\si{\\ampere}$} & {$U$ in $\si{\\volt}$} ', [1, 1],
caption = 'Zinkprobe: Messung der Spannung in Abhängigkeit vom Strom ', label = 'tab: uri_zink')



I_kupfer_raw, U_kupfer_raw = np.genfromtxt('uri_kupfer.txt', unpack=True)
I_kupfer = Q_(I_kupfer_raw, 'ampere')
U_kupfer = Q_(U_kupfer_raw, 'millivolt')
params_kupfer_R, cov_kupfer_R = curve_fit(F_1, I_kupfer.magnitude, U_kupfer.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(-0.2, 10.2, 100)
plt.plot(I_kupfer.magnitude, U_kupfer.magnitude, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_kupfer_R), '-b', label='Lineare Regression')
plt.xlim(I_kupfer.magnitude[0], I_kupfer.magnitude[-1])
plt.xlabel('$I$ in $A$')
plt.ylabel('$U$ in $mV$')
plt.xlim(I_lim[0], I_lim[-1])
plt.grid()
plt.legend(loc='best')
plt.savefig('uri_kupfer.pdf')
R_kupfer_errors = np.sqrt(np.diag(cov_kupfer_R))
R_kupfer = Q_(ufloat(params_kupfer_R[0], R_kupfer_errors[0]), 'millivolt/ampere').to('milliohm')
#print('R_kupfer = ', R_kupfer)

latex.Latexdocument('uri_kupfer_tab.tex').tabular([I_kupfer_raw, U_kupfer_raw],
'{$I$ in $\si{\\ampere}$} & {$U$ in $\si{\\volt}$} ', [1, 1],
caption = 'Kupferprobe: Messung der Spannung in Abhängigkeit vom Strom ', label = 'tab: uri_kupfer')



#Hallspannung Kupfer
#konst. B_Feld bei I_q = 3A
B_konst_kupfer = B(3)
U_ges_min_kupfer_konstB, U_ges_plu_kupfer_konstB = np.genfromtxt('u_h_konstB_kupfer.txt', unpack=True)
U_h_kupfer_konstB = Q_(0.5 * (U_ges_plu_kupfer_konstB - U_ges_min_kupfer_konstB), 'millivolt')
I = np.linspace(0, 10 , 11)

latex.Latexdocument('u_h_kupfer_konstB_tab.tex').tabular([I, U_ges_min_kupfer_konstB, U_ges_plu_kupfer_konstB, U_h_kupfer_konstB.magnitude],
'{$I$ in $\si{\\ampere}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$}', [1, 3, 3, 3],
caption = 'Hallspannung Kupfer bei konstantem Magnetfeld', label = 'tab: hall_kupfer_konstB')


params_kupfer_U_h_1, cov_kupfer_U_h_1 = curve_fit(F_1, I, U_h_kupfer_konstB.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(-0.2, 10.2, 100)
plt.plot(I, U_h_kupfer_konstB, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_kupfer_U_h_1), '-b', label='Lineare Regression')
plt.xlim(I_lim[0], I_lim[-1])
plt.xlabel('$I$ in $A$')
plt.ylabel('$U_H$ in $mV$')
plt.grid()
plt.legend(loc='best')
plt.savefig('u_h_kupfer_konstB.pdf')

Steigung_U_h_kupfer_konstB_errors = np.sqrt(np.diag(cov_kupfer_U_h_1))
#print(params_kupfer_U_h_1[0])
Steigung_U_h_kupfer_konstB = Q_(ufloat(params_kupfer_U_h_1[0], Steigung_U_h_kupfer_konstB_errors[0]), 'millivolt/ampere')
#print('Steigung der Hall Spannung, Kupfer, konst BFeld: ', Steigung_U_h_kupfer_konstB.to('millivolt/ampere'))
n_kupfer_konstB =  (- 1/(Steigung_U_h_kupfer_konstB * e_0 * d_kupfer) * B_konst_kupfer).to('1/meter**3')
#print('n_kupfer_konstB', n_kupfer_konstB)
#print('Fermienergie Kupfer: ', E_fermi(n_kupfer_konstB))

B_konst_zink = B(3)
U_ges_min_zink_konstB, U_ges_plu_zink_konstB = np.genfromtxt('u_h_konstB_zink.txt', unpack=True)
U_h_zink_konstB = Q_(0.5 * (U_ges_plu_zink_konstB - U_ges_min_zink_konstB), 'millivolt')
I = np.linspace(0, 10 , 11)
with open('u_h_zink_konstB_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\caption{Hallspannung Zink bei konstantem Magnetfeld} \n \\label{tab: hall_zink_konstB} \n\\begin{tabular}{')
    f.write(4 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I)):
        f.write('{:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I[i], U_ges_min_zink_konstB[i], U_ges_plu_zink_konstB[i], U_h_zink_konstB[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n  \\end{table}')


params_zink_U_h_1, cov_zink_U_h_1 = curve_fit(F_1, I, U_h_zink_konstB.magnitude, sigma=0.1)
plt.clf()
I_lim = np.linspace(-0.2, 10.2, 100)
plt.plot(I, U_h_zink_konstB, 'rx', label='Messwerte')
plt.plot(I_lim, F_1(I_lim, *params_zink_U_h_1), '-b', label='Lineare Regression')
plt.xlim(I_lim[0], I_lim[-1])
plt.xlabel('$I$ in $A$')
plt.ylabel('$U_H$ in $mV$')
plt.grid()

plt.legend(loc='best')
plt.savefig('u_h_zink_konstB.pdf')

Steigung_U_h_zink_konstB_errors = np.sqrt(np.diag(cov_zink_U_h_1))
Steigung_U_h_zink_konstB = Q_(ufloat(params_zink_U_h_1[0], Steigung_U_h_zink_konstB_errors[0]), 'millivolt/ampere')
#print('Steigung der Hall Spannung, Zink, konst BFeld: ', Steigung_U_h_zink_konstB.to('millivolt/ampere'))
n_zink_konstB =   (1/(Steigung_U_h_zink_konstB * e_0 * d_zink) * B_konst_zink).to('1/meter**3')
#print('n_zink_konstB', n_zink_konstB)
#print('Fermienergie Zink: ', E_fermi(n_zink_konstB))



#Berechnung UH variables B Feld
konstI = Q_(10, 'ampere')
I_konstI, U_ges_plu_zink_konstI, U_ges_min_zink_konstI = np.genfromtxt('u_h_konstI_zink.txt', unpack=True)
U_h_zink_konstI = Q_(0.5 * (U_ges_plu_zink_konstI - U_ges_min_zink_konstI), 'millivolt')

with open('u_h_zink_konstI_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n\\caption{Hallspannung Zink bei konstantem Querstrom} \n \\label{tab: hall_zink_konstI} \n \\begin{tabular}{')
    f.write(5 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$B$ in $\si{\milli\\tesla}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,len(I_konstI)):
        f.write('{:.1f} & {:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I_konstI[i],B(I_konstI[i]).magnitude ,U_ges_min_zink_konstI[i], U_ges_plu_zink_konstI[i], U_h_zink_konstI[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n   \\end{table}')


B_konstI_zink = B(I_konstI).magnitude
params_zink_U_h_2, cov_zink_U_h_2 = curve_fit(F_1, B_konstI_zink[:-1], U_h_zink_konstI.magnitude[:-1], sigma=0.1)
plt.clf()
B_lim = np.linspace(B_konstI_zink[0]+10, B_konstI_zink[-1]-10, 100)
plt.plot(B_konstI_zink, U_h_zink_konstI.magnitude, 'rx', label='Messwerte')
plt.plot(B_lim, F_1(B_lim, *params_zink_U_h_2), '-b', label='Lineare Regression')
plt.xlabel('$B$ in $mT$')
plt.ylabel('$U_H$ in $mV$')
plt.xlim(B_konstI_zink[-1]-10, B_konstI_zink[0]+10)
plt.grid()
plt.legend(loc ='best')
plt.savefig('u_h_zink_konstI.pdf')

Steigung_U_h_zink_konstI_errors = np.sqrt(np.diag(cov_zink_U_h_2))
Steigung_U_h_zink_konstI = Q_(ufloat(params_zink_U_h_2[0], Steigung_U_h_zink_konstI_errors[0]), 'volt/tesla')
#print('Steigung der Hall Spannung, Zink, konst Strom: ', Steigung_U_h_zink_konstI.to('volt/tesla'))
n_zink_konstI =   (1/(Steigung_U_h_zink_konstI * e_0 * d_zink) * konstI).to('1/meter**3')
#print('n_zink_konstI', n_zink_konstI)


konstI = Q_(10, 'ampere')
I_konstI, U_ges_plu_kupfer_konstI, U_ges_min_kupfer_konstI = np.genfromtxt('u_h_konstI_kupfer.txt', unpack=True)
U_h_kupfer_konstI = Q_(0.5 * (U_ges_plu_kupfer_konstI - U_ges_min_kupfer_konstI), 'millivolt')
with open('u_h_kupfer_konstI_tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n\\caption{Hallspannung Kupfer bei konstantem Querstrom} \n \\label{tab: hall_kupfer_konstI} \n \\begin{tabular}{')
    f.write(5 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {$I$ in $\si{\\ampere}$} & {$B$ in $\si{\milli\\tesla}$} & {$U_{ges-}$ in $\si{\milli \\volt}$}  &  {$U_{ges+}$ in $\si{\milli \\volt}$} & {$U_{H}$ in $\si{\milli \\volt}$} \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0, len(I_konstI)):
        f.write('{:.1f} & {:.1f} & {:.3f} & {:.3f} & {:.3f} \\\ \n'.format(I_konstI[i] ,B(I_konstI[i]).magnitude ,U_ges_min_kupfer_konstI[i], U_ges_plu_kupfer_konstI[i], U_h_kupfer_konstI[i].magnitude))
    f.write('\\bottomrule \n \\end{tabular} \n   \\end{table}')

B_konstI_kupfer = B(I_konstI).magnitude
params_kupfer_U_h_2, cov_kupfer_U_h_2 = curve_fit(F_1, B_konstI_kupfer[1:], U_h_kupfer_konstI.magnitude[1:], sigma=0.1)
plt.clf()
B_lim = np.linspace(B_konstI_kupfer[0]+10, B_konstI_kupfer[-1]-10, 100)
plt.plot(B_konstI_kupfer, U_h_kupfer_konstI.magnitude, 'rx', label='Messwerte')
plt.plot(B_lim, F_1(B_lim, *params_kupfer_U_h_2), '-b', label='Lineare Regression')
plt.xlabel('$B$ in $mT$')
plt.ylabel('$U_H$ in $mV$')
plt.xlim(B_konstI_kupfer[-1]-10, B_konstI_kupfer[0]+10)
plt.grid()
plt.legend(loc ='best')
plt.savefig('u_h_kupfer_konstI.pdf')

Steigung_U_h_kupfer_konstI_errors = np.sqrt(np.diag(cov_kupfer_U_h_2))
Steigung_U_h_kupfer_konstI = Q_(ufloat(params_kupfer_U_h_2[0], Steigung_U_h_kupfer_konstI_errors[0]), 'volt/tesla')
#print('Steigung der Hall Spannung, Kupfer, konst Strom: ', Steigung_U_h_kupfer_konstI.to('volt/tesla'))
n_kupfer_konstI =  (- 1/(Steigung_U_h_kupfer_konstI * e_0 * d_kupfer) * konstI).to('1/meter**3')
#print('n_kupfer_konstI', n_kupfer_konstI)

#Berechnungen weiterer Größen
rho_kupfer = Q_(8.96, 'gram/(cm)^3').to('kilogram/m^3')
rho_zink = Q_(7.14, 'gram/(cm)^3').to('kilogram/m^3')
molmass_kupfer = Q_(63.5, 'gram/mol').to('kilogram/mol')
molmass_zink = Q_(65.4, 'gram/mol').to('kilogram/mol')

molvol_kupfer = molmass_kupfer/rho_kupfer
molvol_zink = molmass_zink/rho_zink
vol = Q_(1, 'meter^3')
#print(molvol_zink, molvol_kupfer)
n_cube_kupfer = vol/molvol_kupfer
n_cube_zink = vol/molvol_zink
#print(n_cube_zink, n_cube_kupfer)

z_kupfer_konstI =  (n_kupfer_konstI*(molvol_kupfer / Q_(const.Avogadro, '1/mole')))
#print('z_kupfer_konstI: ', z_kupfer_konstI)
z_kupfer_konstB =  (n_kupfer_konstB*(molvol_kupfer / Q_(const.Avogadro, '1/mole')))
#print('z_kupfer_konstB: ', z_kupfer_konstB)

z_zink_konstI =  (n_zink_konstI*(molvol_zink / Q_(const.Avogadro, '1/mole')))
#print('z_zink_konstI: ', z_zink_konstI)
z_zink_konstB =  (n_zink_konstB*(molvol_zink / Q_(const.Avogadro, '1/mole')))
#print('z_zink_konstB: ', z_zink_konstB)


#spezifische Leitfähigkeit
R_spez_kupfer = (R_kupfer * b_kupfer * d_kupfer) / l_kupfer
#print('spezifischer Widerstand Kupfer: ', R_spez_kupfer.to('ohm * millimeter^2 / meter'))
R_spez_zink = (R_zink * b_zink * d_zink) / l_zink
#print('spezifischer Widerstand Zink: ',R_spez_zink.to('ohm * millimeter^2 / meter'))

tau_kupfer1 = (2 * m_0) / (n_kupfer_konstB * R_spez_kupfer * e_0**2)
#print(tau_kupfer1.to('second'))
tau_kupfer2 = (2 * m_0) / (n_kupfer_konstI * R_spez_kupfer * e_0**2)
#print(tau_kupfer2.to('second'))

tau_zink1 = (2 * m_0) / (n_zink_konstB * R_spez_zink * e_0**2)
#print(tau_zink1.to('second'))
tau_zink2 = (2 * m_0) / (n_zink_konstI * R_spez_zink * e_0**2)
#print(tau_zink2.to('second'))


j = Q_(1, 'ampere/(millimeter)^2')
v_d_kupfer1 = j / (n_kupfer_konstB * e_0)
#print('v_d_kupfer1: ', v_d_kupfer1.to('millimeter/second'))
v_d_kupfer2 = j / (n_kupfer_konstI * e_0)
#print('v_d_kupfer2: ', v_d_kupfer2.to('millimeter/second'))

v_d_zink1 = j / (n_zink_konstB * e_0)
#print('v_d_zink1: ', v_d_zink1.to('millimeter/second'))
v_d_zink2 = j / (n_zink_konstI * e_0)
#print('v_d_zink2: ', v_d_zink2.to('millimeter/second'))

#print('E_fermi_kupfer1:', E_fermi(n_kupfer_konstB) )
#print('E_fermi_kupfer2:', E_fermi(n_kupfer_konstI) )
#print('E_fermi_zink1:', E_fermi(n_zink_konstB) )
#print('E_fermi_zink2:', E_fermi(n_zink_konstI) )

#print('vt1_kupfer: ', ((2 * E_fermi(n_kupfer_konstB) / m_0)**0.5).to('meter/second'))
#print('vt2_kupfer: ', ((2 * E_fermi(n_kupfer_konstI) / m_0)**0.5).to('meter/second'))
#print('vt1_zink: ', ((2 * E_fermi(n_zink_konstB) / m_0)**0.5 ).to('meter/second'))
#print('vt2_zink: ', ((2 * E_fermi(n_zink_konstI) / m_0)**0.5 ).to('meter/second'))


#print('l1_kupfer: ', (tau_kupfer1 * (2 * E_fermi(n_kupfer_konstB) / m_0)**0.5).to('micrometer'))
#print('l2_kupfer: ', (tau_kupfer2 * (2 * E_fermi(n_kupfer_konstI) / m_0)**0.5).to('micrometer'))
#print('l1_zink: ', (tau_zink1 * (2 * E_fermi(n_zink_konstB) / m_0)**0.5 ).to('micrometer'))
#print('l2_zink: ', (tau_zink2 * (2 * E_fermi(n_zink_konstI) / m_0)**0.5 ).to('micrometer'))


#print('mu_kupfer1:', (0.5 * tau_kupfer1* e_0/m_0).to('meter^2 / (volt * second)') )
#print('mu_kupfer2:', (0.5 * tau_kupfer2* e_0/m_0).to('meter^2 / (volt * second)') )
#print('mu_zink1:', (0.5 * tau_zink1* e_0/m_0).to('meter^2 / (volt * second)') )
#print('mu_zink2:', (0.5 * tau_zink2* e_0/m_0).to('meter^2 / (volt * second)') )

R_spez_kupfer_lit = Q_(0.017e-06, ' ohm * meter').to('ohm * millimeter^2 / meter')
#print ('Literaturwert spezifischer Widerstand Kupfer', R_spez_kupfer_lit)
#print('Prozentuale Abweichung: ', R_spez_kupfer/R_spez_kupfer_lit - 1)

R_spez_zink_lit = Q_(0.059e-06, ' ohm * meter').to('ohm * millimeter^2 / meter')
#print ('Literaturwert spezifischer Widerstand Zink', R_spez_zink_lit)
#print('Prozentuale Abweichung: ', R_spez_zink/R_spez_zink_lit - 1)
