import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
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
	abweichung_des_mittelwertes=1/((len(messreihe))**0.5)*np.std(messreihe)
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
##Versuchskonstanten
c=Q_(ufloat(0.7932 ,0),' nanofarad')
c_sp=Q_(ufloat(0.028,0),'nanofarad')
l=Q_(ufloat(23.954,0),'millihenry')

##

def v_min (l,c,c_k):
	return 1/(2*np.pi*(l*(1/c+2/c_k)**(-1))**0.5)
def v_plu (l,c):
	return 1/(2*np.pi*(l*c)**0.5)
##Aufgabenteil a

c_k,n=np.genfromtxt('teilaufgabe_a_schwingungsmaxima.txt',unpack=True)
c_k=Q_(unp.uarray(c_k,c_k*0.2),'nanofarad')
n=unp.uarray(n,1)
print(type(n))
#Verhältnis
verhaeltnis=1/n
print('Verhaeltnis Schwebung zu Schwingung', verhaeltnis)
print('\n')

latex.Latexdocument('teila_ck_n.tex').tabular([	c_k.magnitude,n,verhaeltnis],'{$C_k$ in $\si{\\nano\\farad}$$} & {Anzahl der Schwingungsmaxima} & {Verhältnis}',[1,1,1],
	caption=' Anzahl der Schwingungsmaxima bei verschiedenenen Kapazitäten $C_k$', label='teila_n_ck')


#Bestimmung der Schwingungsfrequenzen
v_mint=v_min(l,c,c_k).to('kilohertz')
v_plut=v_plu(l,c).to('kilohertz')
print('Schwingungsfrequenz v_-',v_mint)
print('\n')
print('Schwingungsfrequenz v_+',v_plut)
print('\n')



##Aufgabenteil b
c_k_1,v_plug,v_ming=np.genfromtxt('teilaufgabe_b_frequenzen.txt',unpack=True)
v_ming=Q_(unp.uarray(v_ming,0.06),'kilohertz')
v_plug=Q_(unp.uarray(v_plug,0.06),'kilohertz')

##Verhältniss Theorie und Praxis
v_min_verhael=v_ming/v_mint
v_plu_verhael=v_plug/v_plut

print('Verhältnis von v_m',v_min_verhael)
print('Verhältnis von v_+',v_plu_verhael)
print('\n')

latex.Latexdocument('teilb_schwingungen_prak_theo.tex').tabular([c_k.magnitude,v_ming.magnitude,v_plug.magnitude,v_min_verhael.magnitude,v_plu_verhael.magnitude],
	'{$C_k in $\si{\\nano\\farad}$$} & {$Schwingungsfrequenz $\\nu_-$ in $\si{\kilo\hertz}$$} & {$Schwingungsfrequenz $\\nu_+$ in $\si{\kilo\hertz}$$ }& {$Verhältnis $\\nu_-$$ & $Verhältnis $\\nu_+$$}',[1,1,1,1,1],
	caption=' Bestimme Fundamentalfrequenzen mit den Verhältnis zur Theorie', label='teilb_schwingungen_prak_theo')


v_plu_verhael_mittel=(sum(v_plu_verhael)/len(v_plu_verhael))
print('Gemittelte Abweichung v_plu',v_plu_verhael_mittel)
print('\n')

##Aufgabenteil c

periode=Q_(1,'second')
startf=Q_(15.67,'kilohertz')
endf=Q_(96.15,'kilohertz')
m=(endf-startf)/periode
print('Steigung m',m)
print('\n')

def zeit_f_gerade(t):
	return m*t+startf

c_k,t_1,t_2=np.genfromtxt('teilaufgabe_c_deltaT.txt',unpack=True)
t_1=Q_(unp.uarray(t_1,5),'millisecond')
t_2=Q_(unp.uarray(t_2,5),'millisecond')


frequenzen_t1=zeit_f_gerade(t_1).to('kilohertz')
frequenzen_t2=zeit_f_gerade(t_2).to('kilohertz')
print('Frequenz Teil c v+',frequenzen_t1)
print('\n')
v_pu_verhael_c=frequenzen_t1/v_plut
v_plu_verhael_mittel_c=(sum(v_pu_verhael_c)/len(v_pu_verhael_c))

print('Verhältnis Teilc v+, gemittelt', v_pu_verhael_c,v_plu_verhael_mittel_c)
print('\n')
print('Frequenz Teil c v-',frequenzen_t2)
print('\n')
v_min_verhael_c=frequenzen_t2/v_mint
print('Verhältnis Teilc v-', v_min_verhael_c)
print('\n')





#Plotbereich

#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()
#
#plt.plot(,,'rx',label='')
#
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('.pdf')
