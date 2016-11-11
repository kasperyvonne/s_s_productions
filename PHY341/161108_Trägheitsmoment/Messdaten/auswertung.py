import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Hier nicht wichtig


abstand, schwingungsdauer = np.genfromtxt('eigenmoment.txt',unpack=True)
winkelrecht_passiv_wert,winkelrecht_passiv_fehler =np.genfromtxt('Winkelrichtgröße_passiv.txt',unpack=True)
winkelrecht_passiv=ufloat(winkelrecht_passiv_wert,winkelrecht_passiv_fehler)
schingungsdauer_grauer_zylinder=np.genfromtxt('T_zylinder_gross.txt',unpack=True)
schwingungsdauer_kugel=np.genfromtxt('T_grosse_kugel.txt',unpack=True)

#Berechnung von  D pasiv:

#def D(rad, wink, kraf):
#	return (kraf*rad)/wink
#
#wi_rad=[]
#for w in wi:
#	wi_rad.append(math.radians(w))
#print(wi_rad)
#winkelricht=[]
#while n<len(abst):
#	print('Bin dirn')
#	winkelricht.append( D(abst[n],wi_rad[n],kra[n]))
#	n+=1
#print(winkelricht)
#winkelricht_mitt=sum(winkelricht)/len(winkelricht)
#winkelricht_abweichung_mitt= 1/(np.sqrt(len(winkelricht)))*np.std(winkelricht)
#
#winkelrecht_passiv=ufloat(winkelricht_mitt,winkelricht_abweichung_mitt)
#
#print(winkelrecht_passiv)
#
#print(winkelrecht_passiv.n)

#Berehnung von D dynamisch:
hoehe_zylinder=0.5*3.49
abstand+=hoehe_zylinder
abstand*=10e-2
schwingungsdauer*=(1/5)
schwingungsdauer_hilf=[]

#print(abstand[::3])
for i in range(len(schwingungsdauer))[::3]:
    schwingungsdauer_hilf.append(np.mean(schwingungsdauer[i:i+3]))
schwingungsdauer_mittel=np.array(schwingungsdauer_hilf)
#print('Zeitlichesmittel ', schwingungsdauer_mittel**2)


def f(m,u,b):
	return m*u+b

params_p,covarian=curve_fit(f,abstand[::3]**2,schwingungsdauer_mittel**2)

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


m,m_e,b,b_e=linregress(abstand[::3]**2,schwingungsdauer_mittel**2)

m_fehler=ufloat(m,m_e)
b_fehler=ufloat(b,b_e)
print('Steigung ', m_fehler)
m_1=222.51e-3
m_2=223.46e-3
winkelrichtgroesse=(((m_1+m_2)*4*np.pi**2)/m_fehler)
print('Winkelrichtgroesse ', winkelrichtgroesse)



#Berechnung vom Eigenträgheitsmoment

def J_zylinder(radius,hoehe,masse):
	return (1/4)*masse*radius**2+(1/12)*masse*hoehe**2

radius_zylinder=(3.49*0.5)*10**(-3)
hoehe_zylinder=(14.05*0.5)*10**(-3)
J_zylinder1=J_zylinder(radius_zylinder,hoehe_zylinder,m_1)
J_zylinder2=J_zylinder(radius_zylinder,hoehe_zylinder,m_2)

print(J_zylinder1)


#Berechnung Trägheitsmoment Zylinder grau
schingungsdauer_grauer_zylinder*=(1/5)
mittel_schingungsdauer_grauer_zylinder=sum(schingungsdauer_grauer_zylinder)/len(schingungsdauer_grauer_zylinder)
mittel_schingungsdauer_grauer_zylinder_fehler= 1/(np.sqrt(len(schingungsdauer_grauer_zylinder)))*np.std(winkelricht)
schingungsdauer_grauer_zylinder_u=ufloat(mittel_schingungsdauer_grauer_zylinder,mittel_schingungsdauer_grauer_zylinder_fehler)
J_eigen=-(J_zylinder1+J_zylinder2)+(winkelrecht_passiv)/(4*np.pi**2)*b_fehler







#Plotbereich
plt.xlim(0,6.4)
plt.ylim(0,52)
x=np.linspace(0,6.5,1000)

plt.plot(abstand[::3]**2,schwingungsdauer_mittel**2,'rx',label='Messwerte')
plt.plot(x**2,m*x**2+b,'b-',label='Lineare Regression')
plt.plot(abstand[::3]**2,f(abstand[::3]**2,*params_p),'y-',label='Fit')
plt.legend(loc='best')
plt.ylabel(r'$T^2 \ in \ \frac{1}{\mathrm{s}^2}$')
plt.xlabel(r'$r^2 \ in  \ \mathrm{m}^2$')
plt.grid()
#plt.savefig('lineare_regression.pdf')
plt.show()



#np.savetxt('Winkelrichtgröße_passiv.txt', np.column_stack([winkelrecht_passiv.n,winkelrecht_passiv.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')