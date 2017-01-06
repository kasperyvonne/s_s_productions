import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry

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

def v_min (l,c,c_k):
	return 1/(2*np.pi*(l*(1/c+2/c_k)**(-1))**0.5)
def v_plu (l,c):
	return 1/(2*np.pi*(l*c)**0.5)
##Aufgabenteil a


c=Q_(ufloat( ,),'farad')
c_k=Q_(ufloat(,),'farad')
l=Q_(ufloat(,),'henry')

#Bestimmung der Schwingungsfrequenzen
v_mint=v_min(l,c,c_k).to('1/second')
v_plut=v_plu(l,c).to('1/second')
print('Schwingungsfrequenz v_-',v_min)
print('Schwingungsfrequenz v_+',v_plu)
print('\n')

#Bestimmungen der Verhältnisse


##Aufgabenteil b
v_ming,v_plug=np.genfromtxt('.txt',unpack=True)
v_ming=Q_(v_ming,'1/second')
v_plug=Q_(v_plug,'1/second')

#Verhältniss Theorie und Praxis
v_min_verhael=v_ming/v_mint
v_plu_verhael=v_plug/v_plut

print('Verhältnis von v_m',v_min_verhael)
print('Verhältnis von v_+',v_plu_verhael)
print('\n')

##Aufgabenteil c

def I_2 (u,omega,c_k,l,c,r):
	omega*=2*np.pi
	z=omega*l-1/omega*(1/c+1/c_k)
	i_2=u/(4*omega**2*c_k**2*r**2*z**2+(1/(omega*c_k)-omega*c_k*z**2+omega*r**2*c_k)**2)**0.5
	return i_2

def I_2u (u,r):
	return u/r

#Bestimmung von I_2
r_min=Q_(85,'ohm')
r_plu=Q_(73,'ohm')

I_2_mint=I_2()
I_2_plut=I_2()
I_2_ming=I_2u(,r_min)
I_2_plug=I_2u(,r_plu)

print('I_2+ theoretisch', I_2_plut)
print('I_2+ praktisch', I_2_plug)
print('I_2- theoretisch', I_2_mint)
print('I_2- praktisch', I_2_ming)
print('\n')





#Bestimmung von I_k



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
