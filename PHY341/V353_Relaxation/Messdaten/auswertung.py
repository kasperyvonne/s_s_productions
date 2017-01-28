import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex

u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'ohm')
c = Q_(0, 'degC')
c.to('kelvin')
#print(c.to('kelvin'))
#print(a**2)
#print(b**2)
#einheitentst=Q_(1*1e-3,'farad')
#einheitentst_2=Q_(1,'ohm')
#print(einheitentst)
#print(1/(einheitentst*einheitentst_2).to('second'))


#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

#Standartabweichung und Mittelwert

def mittel_und_abweichung(messreihe):
	messreihe_einheit=messreihe.units
	mittelwert=sum(messreihe)/len(messreihe)
	abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe)))*np.std(messreihe)
	mittel_und_abweichung=Q_(unp.uarray(mittelwert,abweichung_des_mittelwertes),messreihe_einheit)
	return mittel_und_abweichung

#Standartabweichung und Mittelwert für Messreihe mit Intervallen
def mittel_und_abweichung_intervall(messreihe,intervall_laenge):
	messreihe_einheit=messreihe.units
	mittelwert_abweichung_liste=[]
	for i in range(len(messreihe))[::intervall_laenge]:
		mittelwert=sum(messreihe[i:i+intervall_laenge])/len(messreihe[i:i+intervall_laenge])
		abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe[i:i+intervall_laenge])))*np.std(messreihe[i:i+intervall_laenge])
		mittelwert_abweichung_liste.append(ufloat(mittelwert.magnitude,abweichung_des_mittelwertes.magnitude))
	mittelwert_abweichung_u=Q_(unp.uarray(unp.nominal_values(mittelwert_abweichung_liste),unp.std_devs(mittelwert_abweichung_liste)),messreihe_einheit)
	return mittelwert_abweichung_u


#Lineare regression

def linregress(x, y):
    assert len(x) == len(y)

    x, y = np.array(x), np.array(y)

    N = len(y)
    Delta = N * np.sum(x**2) - (np.sum(x))**2

    # A ist die Steigung, B der y-Achsenabschnitt
    A = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)

    return A, A_error, B, B_error


#Angepasstes Programm

##teila

t,u_a=np.genfromtxt('eingabe_teila_neu.txt',unpack=True)
t*=1e-3

def g (x,a,b,c):
	return a*np.exp(b*x)+c

params_a,covariance_a=curve_fit(g,t,u_a)
error_params_a=np.sqrt(np.diag(covariance_a))
print(params_a)
print(error_params_a)
params_a_u=ufloat(params_a[1],error_params_a[1])
print('RC in Teil a',1/params_a_u)
print('\n')

def ger (x,m,b):
	return m*x+b

params_a2,covariance_a2=curve_fit(ger,t,(u_a))
error_params_a2=np.sqrt(np.diag(covariance_a2))
print(params_a2)
print(error_params_a2)
params_a_u=ufloat(params_a2[0],error_params_a2[1])
print('RC in Teil a, linearer Fit',1/params_a_u)
print('\n')

latex.Latexdocument('abgelesene_werte_teila.tex').tabular([t, u_a],
r'{t in $\si{\milli\second}$} & {U\ua{ein} in $\si{\volt}$}', [1, 1],
caption = r'Aus der Grafik \ref{} bestimmte Messwerte', label = 'tab:teil_a_spannungen')

##teilb

f, u_c, u_g = np.genfromtxt('u_c_und_u_g.txt',unpack=True)
f= f*u.hertz
u_c= 0.5*u_c* u.volt
u_g =0.5*u_g* u.volt

u_c_normierung=u_c/u_g
print('Verhältnis Uc/Ug', u_c_normierung)
print('\n')

def fit(x,a):
	return 1/(np.sqrt(1+(2*np.pi*x)**2*a**2))

params_b,covariance_b=curve_fit(fit,f.magnitude,u_c_normierung.magnitude)
error_params_b=np.sqrt(np.diag(covariance_b))
params_b_u=ufloat(params_b,error_params_b)

print('Paramter RC aus der Messung b',params_b_u)
print('\n')

latex.Latexdocument('teil_b.tex').tabular([f, u_g,u_c,u_c_normierung],
r'{f in $\si{\hertz}$} & {U\ua{g} in $\si{\volt}$}& {U\ua{c} in $\si{\volt}$} & {\frac{U\ua{c}{U\ua{g}}$}', [1, 1, 1, 1],
caption = r'Gemessene Generator- und Kondensatorspannungen bei unterschiedlichen Frequenzen ', label = 'tab:teil_b_spannungen')

#teilc
f_c,a,b=np.genfromtxt('abstand.txt',unpack=True)
a=a*1e-6*u.second
b=b*u.millisecond
print(b.to('second'))

def phi(a,b):
	return (a/b.to('second'))*2*np.pi
phase=phi(a,b)

print('Phase',phase)
print('\n')

def phi_fit (x,a):
	return np.arctan(-(2*np.pi*x)*a)

params_c,covariance_c=curve_fit(phi_fit,f_c,phase)
error_params_c=np.sqrt(np.diag(covariance_c))
params_c_u=ufloat(params_c,error_params_c)

latex.Latexdocument('teil_b.tex').tabular([f_c,a,b],
r'{f in $\si{\hertz}$} & {a in $\si{\second}$}& {b in $\si{\milli\second}$}', [1, 1, 1],
caption = r'Gemessene Abstände $a$ und $b$ ', label = 'tab:teil_c_abstände')

#Plotbereich
##Plots zu b


plt.clf()
#loga
plt.xlim(8,f[-1].magnitude+1000)
plt.ylim(0,max(u_c_normierung)+0.025)
aufvariabele=np.linspace(f[0].magnitude-10,f[-1].magnitude+1000,10000)
plt.plot(f.magnitude,u_c_normierung,'rx',label=r'$\mathrm{berechnete} \, \mathrm{Werte}$')
plt.plot(aufvariabele,fit(aufvariabele,*params_b),'b-',label=r'$\mathrm{Fit}$')
plt.grid()
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel(r'$\mathrm{Frequenz}\, \mathrm{in} \,\mathrm{Hz}$')
plt.ylabel(r'$ \mathrm{Verhältnis} \,\, \frac{U_c}{U_g}$')
#plt.show()
#plt.savefig('u_cdurchu_g.pdf')

##Plots zu c
plt.clf()
plt.xlim(8,f_c[-1]+1000)
plt.ylim(0,max(phase)+0.025)
laufvariabele=np.linspace(f_c[0]-10,f_c[-1]+1000,1000)
plt.plot(f_c,phase,'bx',label=r'$\mathrm{Phase}$')
plt.plot(laufvariabele,phi_fit(*params_c,laufvariabele),'r-',label=r'$\mathrm{Fit}$')
plt.xscale('log')
plt.legend(loc='best')
plt.xlabel(r'$\nu \, \mathrm{in} \,\mathrm{Hz}$')
plt.ylabel(r'$ \mathrm{Verhältnis} \,\, \frac{U_c}{U_g}$')
plt.yticks([0,1/16*np.pi,1/8*np.pi,3/16*np.pi,1/4*np.pi,5/16*np.pi,3/8*np.pi,7/16*np.pi,1/2*np.pi,9/16*np.pi,5/8*np.pi,11/16*np.pi,3/4*np.pi,13/16*np.pi],
['0','$\\frac{1}{16}\\pi$', '$\\frac{1}{8}\\pi$','$\\frac{3}{16}\\pi$' ,'$\\frac{1}{4}\\pi$','$\\frac{5}{16}\\pi$','$\\frac{3}{8}\\pi$','$\\frac{7}{16}\\pi$','$\\frac{1}{2}\\pi$','$\\frac{9}{16}\\pi$','$\\frac{5}{8}\\pi$','$\\frac{11}{16}\\pi$','$\\frac{3}{4}\\pi$','$\\frac{13}{16}\\pi$'])
plt.grid()
#plt.show()
#plt.savefig('frequenz_phase.pdf')

##Plots zu d
plt.clf()

plt.polar(phase,fit(f_c,*params_b) ,'rx',label=r'$\mathrm{Messdaten}$')
winkel=np.linspace(0,phase[-1],1000)
plt.polar(winkel,np.cos(winkel),'b-',label=r'$\mathrm{Theoriekurve}$')
plt.xticks([0,0.25*np.pi,0.5*np.pi,0.75*np.pi,np.pi,1.25*np.pi,1.5*np.pi,1.75*np.pi],['0','$\\frac{1}{4}\\pi$', '$\\frac{1}{2}\\pi$','$\\frac{3}{4}\\pi$' ,'$\\pi$','$\\frac{5}{4}\\pi$','$\\frac{3}{2}\\pi$','$\\frac{7}{4}\\pi$'])
plt.legend(loc=[0.05,0.95])
#plt.show()
#plt.savefig('polarplot.pdf')

plt.clf()
plt.xlim(t[0]-1e-2,t[-1]+1e-2)
plt.ylim(0.67,10)
plt.plot(t,u_a,'bx',label=r'$\mathrm{Messwerte}$')
plt.plot(t,g(t,*params_a),'r-',label=r'$\mathrm{Fit}$')
plt.legend(loc='best')
plt.xlabel(r'$t \, \mathrm{in} \, ms$')
plt.ylabel(r'$ U_{\mathrm{ent}} \, \mathrm{in} \, V$')
plt.yscale('log')
plt.grid()
#plt.show()
plt.savefig('teil_a_entladung.pdf')
