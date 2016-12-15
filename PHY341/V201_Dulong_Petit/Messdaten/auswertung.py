import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from pint import UnitRegistry
import scipy.constants as const
import matplotlib.pyplot as plt
u = UnitRegistry()
Q_ = u.Quantity

R = Q_(const.gas_constant, 'joule / (mole * kelvin)')
c_wasser = Q_(4.18, 'joule/(gram kelvin)')
rho_wasser = Q_(992.2, 'kg / (m^3)')
m_wasser = Q_(0.6, 'liter').to('m^3') * rho_wasser

#werte für die Berechnung von cgmg
m_wasser_x = Q_(0.3, 'liter').to('m^3') * rho_wasser
m_wasser_y = m_wasser_x
print('Masse_x: ', m_wasser_x)
U_y = Q_(2.760, 'dimensionless')
U_x = Q_(0.749, 'dimensionless')
U_misch_for_cgmg = Q_(1.594, 'dimensionless')

#Ergebnisse aus Eichung der Thermoelemente, bestimmung der Geradensteigung
U_0 = 0
U_100 = 3.98
x = np.linspace(0, 100)
m = 100 / (U_100 - U_0)
print ('Steigung: ', m)
def UtoT(u):
    return Q_(m * u + 273.15, 'kelvin')





#berechnung cm_mg
T_y = UtoT(U_y)
T_x = UtoT(U_x)
print('Temperaturen x, y:  ', T_x, T_y)
T_misch_for_cgmg = UtoT(U_misch_for_cgmg)
c_g_m_g = ((c_wasser * m_wasser_y * (T_y - T_misch_for_cgmg)  -  c_wasser * m_wasser_x*(T_misch_for_cgmg - T_x))/(T_misch_for_cgmg - T_x) ).to('joule/kelvin')
print('c_g_m_g:   ', c_g_m_g)


#einlesen der konstanten für graphit
rho_graphit_raw, M_graphit_raw, alpha_graphit_raw, kappa_graphit_raw = np.genfromtxt('materialkonstanten_graphit.txt', unpack=True)
rho_graphit = Q_(rho_graphit_raw, 'gram/centimeter^3').to('gram/(m^3)')
M_graphit = Q_(M_graphit_raw, 'gram/mol')
alpha_graphit = Q_(alpha_graphit_raw * 1e-06, '1/kelvin')
kappa_graphit = Q_(kappa_graphit_raw * 1e09, 'kilogram/(meter*second**2)')
mol_vol_graphit = M_graphit / rho_graphit
#print(mol_vol_graphit)

#Einlesen der Werte für Graphit
m_k_graphit = Q_(247.79 - 139.77, 'gram').to('kilogram')
U_k_graphit, U_w_graphit, U_m_graphit = np.genfromtxt('graphit.txt', unpack=True)
T_k_graphit = UtoT(U_k_graphit)
T_w_graphit = UtoT(U_w_graphit)
T_m_graphit = UtoT(U_m_graphit)
print('TK, TW, TM Graphit:  ', T_k_graphit, T_w_graphit, T_m_graphit)

#berechnung von c_k
c_k_graphit = ((c_wasser * m_wasser + c_g_m_g)*(T_m_graphit - T_w_graphit))/(m_k_graphit*(T_k_graphit - T_m_graphit)) * M_graphit
print('ck Graphit:  ',c_k_graphit)
print(R)
#print('a', (c_k_graphit - 9 * alpha_graphit**2 * kappa_graphit * mol_vol_graphit * T_m_graphit)/R)
const_graphit = (alpha_graphit**2 * kappa_graphit * mol_vol_graphit * T_m_graphit).to('joule/(mole*kelvin)')
c_v_graphit = c_k_graphit - const_graphit
c_graphit_lit =  Q_(0.751, 'joule /(gram* kelvin)' ) * M_graphit
print('Literaturwert der spezifischen Wärmekapazität Graphit: ', c_graphit_lit)
print(c_v_graphit)
print(c_v_graphit /  (3 * R) -1 )
print('Mittelwert Graphit: ', np.mean(c_v_graphit), 'pm', 1/np.sqrt(3) * np.std(c_v_graphit))




#einlesen der konstanten für blei
rho_blei_raw, M_blei_raw, alpha_blei_raw, kappa_blei_raw = np.genfromtxt('materialkonstanten_blei.txt', unpack=True)
rho_blei = Q_(rho_blei_raw, 'gram/centimeter^3').to('gram/(m^3)')
M_blei = Q_(M_blei_raw, 'gram/mol')
alpha_blei = Q_(alpha_blei_raw * 1e-06, '1/kelvin')
kappa_blei = Q_(kappa_blei_raw * 1e09, 'kilogram/(meter*second**2)')
mol_vol_blei = M_blei / rho_blei


#Einlesen der Werte für blei
m_k_blei = Q_(370.53 - 138.5, 'gram').to('kilogram')
U_k_blei, U_w_blei, U_m_blei = np.genfromtxt('blei.txt', unpack=True)
T_k_blei = UtoT(U_k_blei)
T_w_blei = UtoT(U_w_blei)
T_m_blei = UtoT(U_m_blei)
print('TK, TW, TM Zinn:  ', T_k_blei, T_w_blei, T_m_blei)

#berechnung von c_k
c_k_blei = ((c_wasser * m_wasser + c_g_m_g)*(T_m_blei - T_w_blei))/(m_k_blei * (T_k_blei - T_m_blei)) * M_blei
print('spezifische Wärmekapazität Blei:  ', c_k_blei)
const_blei = (alpha_blei**2 * kappa_blei * mol_vol_blei * T_m_blei).to('joule/(mole*kelvin)')
c_blei_lit = Q_(0.230, 'joule /(gram* kelvin)' ) * M_blei
print('Literaturwert der spezifischen Wärmekapazität Zinn: ', c_blei_lit)
c_v_blei = c_k_blei - const_blei
print('Prozentuale Abweichung vom lit Wert', c_v_blei / c_blei_lit  - 1)
print(c_v_blei)
print('Prozentuale Abweichung von 3R: ', c_v_blei /  (3 * R) -1 )
print('Mittelwert zinn: ', np.mean(c_v_blei), 'pm', 1/np.sqrt(3) * np.std(c_v_blei))






#einlesen der konstanten für aluminium
rho_alu_raw, M_alu_raw, alpha_alu_raw, kappa_alu_raw = np.genfromtxt('materialkonstanten_aluminium.txt', unpack=True)
rho_alu = Q_(rho_alu_raw, 'gram/centimeter^3').to('gram/(m^3)')
M_alu = Q_(M_alu_raw, 'gram/mol')
alpha_alu = Q_(alpha_alu_raw * 1e-06, '1/kelvin')
kappa_alu = Q_(kappa_alu_raw * 1e09, 'kilogram/(meter*second**2)')
mol_vol_alu = M_alu / rho_alu


#Einlesen der Werte für alu
m_k_alu = Q_(255.07 - 140.50, 'gram').to('kilogram')
U_k_alu, U_w_alu, U_m_alu = np.genfromtxt('aluminium.txt', unpack=True)
T_k_alu = UtoT(U_k_alu)
T_w_alu = UtoT(U_w_alu)
T_m_alu = UtoT(U_m_alu)
#print('TK, TW, TM Aluminium:  ', T_k_alu, T_w_alu, T_m_alu)


#berechnung von c_k
c_k_alu = ((c_wasser * m_wasser + c_g_m_g)*(T_m_alu - T_w_alu))/(m_k_alu * (T_k_alu - T_m_alu)) * M_alu
print('spezifische Wärmekapazität alu:  ', c_k_alu)
const_alu = (alpha_alu**2 * kappa_alu * mol_vol_alu * T_m_alu).to('joule/(mole*kelvin)')
c_alu_lit = Q_(0.896, 'joule /(gram* kelvin)' ) * M_alu
print('Literaturwert der spezifischen Wärmekapazität Alu: ', c_alu_lit)
c_v_alu = c_k_alu - const_alu
print('Prozentuale Abweichung vom lit Wert', c_v_alu / c_alu_lit  - 1)
print(c_v_alu)





with open('tab.tex', 'w') as f:
    f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
    f.write('l' + 3 *'S ')
    f.write('} \n \\toprule  \n')
    f.write(' {Stoff} & {{$c_k$ in $\si{\joule \per {\kelvin \mol}}$}} & {{$C_V$ in $\si{\joule \per {\kelvin \mol}}$}}  &  {$\\frac{C_V}{R}$}  \\\ \n')
    f.write('\\midrule  \n ')
    for i in range (0,3):
        f.write('{{Graphit}} & {:.2f} & {:.2f} & {:.2f}  \\\ \n'.format(c_k_graphit[i].magnitude, c_v_graphit[i].magnitude, c_v_graphit[i].magnitude/R.magnitude))
    for i in range (0,3):
        f.write('{{Zinn}} & {:.2f} & {:.2f} & {:.2f}  \\\ \n'.format(c_k_blei[i], c_v_blei[i], c_v_blei[i] /R ))
    f.write('{{Aluminium}} & {:.2f} & {:.2f} & {:.2f}  \\\ \n'.format(c_k_alu, c_v_alu, c_v_alu /R))
    f.write('\\bottomrule \n \\end{tabular} \n \\caption{Spezifische Wärmekapazitäten} \n \\label{tab: c_v} \n  \\end{table}')
