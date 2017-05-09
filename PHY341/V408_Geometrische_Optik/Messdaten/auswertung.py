import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values
from uncertainties.unumpy import std_devs
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
####FUNKTIONEN
def bessel(g, b):
    d = g - b
    e = g + b
    return (e**2 - d**2)/(4 * e)

def lin(x, m, b):
    return m * x + b

def linfit(x, y):
    params_raw, cov = curve_fit(lin, x, y)
    return correlated_values(params_raw, cov)



#Gegenstandsgröße in cm
G = ufloat(3, 0.1)

####METHODE 1
method_1_g_raw, method_1_b_raw, method_1_B_raw = np.genfromtxt('method_1.txt', unpack=True)
sort = np.argsort(method_1_g_raw)
#sort_b = np.argsort(method_1_g_raw)
####SORTIEREN
method_1_g = method_1_g_raw[sort]
method_1_b = method_1_b_raw[sort]
method_1_B = method_1_B_raw[sort]

####MITTELWERT BRENNWEITE 1
method_1_f_raw = 1 / (1 / method_1_g + 1 / method_1_b)
method_1_f = ufloat(np.mean(method_1_f_raw), 1/np.sqrt(len(method_1_f_raw) * np.std(method_1_f_raw)))
r.app(r'f\ua{1}', Q_(method_1_f, 'centimeter'))
l.Latexdocument('methode_1.tex').tabular([method_1_g[::-1], method_1_b, method_1_B, method_1_b/method_1_g[::-1], method_1_B[::-1]/G],
header = '{$g/\si{\centi\meter}$} & {$b/\si{\centi\meter}$} & {$B/\si{\centi\meter}$} & {$V_1$} & {$V_1$}', places = [1,1,1,1,1],
label='tab: methode_1', caption='test')

######################


######BESSELMETHODE
bessel_g_1_raw, bessel_b_1_raw, bessel_g_2_raw, bessel_b_2_raw = np.genfromtxt('bessel.txt', unpack=True)
bessel_g_1 = unp.uarray(bessel_g_1_raw, 0.1)
bessel_b_1 = unp.uarray(bessel_b_1_raw, 0.1)
bessel_g_2 = unp.uarray(bessel_g_2_raw, 0.1)
bessel_b_2 = unp.uarray(bessel_b_2_raw, 0.1)
bessel_g = np.concatenate((bessel_g_1, bessel_b_2))
bessel_b = np.concatenate((bessel_b_1, bessel_g_2))
r.app(r'f\ua{2}', Q_(np.mean(bessel(bessel_g, bessel_b)), 'centimeter') )

#######CHROMATISCHE ABBERATION
g_1_blau_raw, b_1_blau_raw, g_2_blau_raw, b_2_blau_raw = np.genfromtxt('blau.txt', unpack=True)
g_1_blau = unp.uarray(g_1_blau_raw, 0.1)
b_1_blau = unp.uarray(b_1_blau_raw, 0.1)
g_2_blau = unp.uarray(g_2_blau_raw, 0.1)
b_2_blau = unp.uarray(b_2_blau_raw, 0.1)
blau_g = np.concatenate((g_1_blau, b_2_blau))
blau_b = np.concatenate((b_1_blau, g_2_blau))

r.app(r'f\ua{b}', Q_(np.mean(bessel(blau_g, blau_b)), 'centimeter') )


g_1_rot_raw, b_1_rot_raw, g_2_rot_raw, b_2_rot_raw = np.genfromtxt('rot.txt', unpack=True)
g_1_rot = unp.uarray(g_1_rot_raw, 0.1)
b_1_rot = unp.uarray(b_1_rot_raw, 0.1)
g_2_rot = unp.uarray(g_2_rot_raw, 0.1)
b_2_rot = unp.uarray(b_2_rot_raw, 0.1)
rot_g = np.concatenate((g_1_rot, b_2_rot))
rot_b = np.concatenate((b_1_rot, g_2_rot))
r.app(r'f\ua{r}', Q_(np.mean(bessel(rot_g, rot_b)), 'centimeter') )


#####ABBE
abbe_g_raw, abbe_b_raw, abbe_B_raw = np.genfromtxt('abbe.txt', unpack=True)
abbe_g = unp.uarray(abbe_g_raw, 0.1)
abbe_b = unp.uarray(abbe_b_raw, 0.1)
abbe_B = unp.uarray(abbe_B_raw, 0.1)
abbe_V = abbe_B/G


x = np.linspace(1.6, 3, 1000)
plt.errorbar(unp.nominal_values(1 + 1/abbe_V), unp.nominal_values(abbe_g),
xerr = unp.std_devs(1 + 1/abbe_V), yerr = unp.std_devs(abbe_g), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
params_g = linfit(unp.nominal_values(1 + 1/abbe_V), unp.nominal_values(abbe_g))
m_g = params_g[0]
r.app(r'f\ua{a, 1}', Q_(m_g, 'centimeter'))
b_g = params_g[1]
r.app('h', Q_(b_g, 'centimeter'))
plt.plot(x, lin(x, m_g.n, b_g.n))
plt.grid()
plt.savefig('plots/abbe_plot_g.pdf')
plt.clf()

plt.errorbar(unp.nominal_values(1 + abbe_V), unp.nominal_values(abbe_b),
xerr = unp.std_devs(1 + abbe_V), yerr = unp.std_devs(abbe_b), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
params_b = linfit(unp.nominal_values(1 + abbe_V), unp.nominal_values(abbe_b))
m_b = params_b[0]
r.app(r'f\ua{a, 2}', Q_(m_b, 'centimeter'))
b_b = params_b[1]
r.app('h-', Q_(b_b, 'centimeter'))
plt.plot(x, lin(x, m_b.n, b_b.n))
plt.grid()
plt.savefig('plots/abbe_plot_b.pdf')
plt.clf()



######WASSERLINSE
wasser_g_raw, wasser_b_raw = np.genfromtxt('wasserlinse.txt', unpack=True)
wasser_f = 1 / (1/wasser_g_raw + 1/wasser_b_raw)
print(np.mean(wasser_f))
for i in range(0, len(wasser_g_raw)):
    plt.plot([wasser_g_raw[i], 0], [0,wasser_b_raw[i]], 'b-')
plt.grid()
plt.savefig('plots/schurz.pdf')
plt.clf()


######PLOTS
######METHODE 1


for i in range(0, 10):
    plt.plot([method_1_g[::-1][i], 0], [0,method_1_b[i]], 'b-')
plt.grid()
plt.axvline(x = method_1_f.n)
plt.xlim(0, 38)
plt.ylim(0, 23)
plt.savefig('plots/methode_1.pdf')



r.makeresults()
