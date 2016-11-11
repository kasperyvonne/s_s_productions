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
schwingungsdauer_puppe_1=np.genfromtxt('T_puppe_postition_1.txt',unpack=True)
schwingungsdauer_puppe_2=np.genfromtxt('T_puppe_position_2.txt',unpack=True)

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

abstand=abstand+hoehe_zylinder

abstand*=1e-2
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
print('y-Achsenabschnitt', b_fehler)
m_1=222.51e-3
m_2=223.46e-3
winkelrichtgroesse_dynamisch=(((m_1+m_2)*4*np.pi**2)/m_fehler)
print('Winkelrichtgroesse-dynamisch ', winkelrichtgroesse_dynamisch)
print('Winkelrichtgröße_passiv', winkelrecht_passiv)



#Berechnung vom Eigenträgheitsmoment

def J_zylinder(radius,hoehe,masse):
	return (1/4)*masse*radius**2+(1/12)*masse*hoehe**2

radius_zylinder=(3.49*0.5)*1e-2
hoehe_zylinder=(14.05*0.5)*1e-2
J_zylinder1=J_zylinder(radius_zylinder,hoehe_zylinder,m_1)
J_zylinder2=J_zylinder(radius_zylinder,hoehe_zylinder,m_2)

J_eigen=-(J_zylinder1+J_zylinder2)+(winkelrichtgroesse_dynamisch)/(4*np.pi**2)*b_fehler

print('Trageheitsmoment_Eigen ', J_eigen)

def traegheitsmoment(schwingungsdauer):
	return (schwingungsdauer**2*winkelrichtgroesse_dynamisch)/(4*(np.pi)**2)

def Mittelwert(schwingungsdauer):
	mittel_schingungsdauer=sum(schwingungsdauer)/len(schwingungsdauer)
	mittel_schingungsdauer_fehler= 1/(np.sqrt(len(schwingungsdauer)))*np.std(schwingungsdauer)
	return ufloat(mittel_schingungsdauer,mittel_schingungsdauer_fehler)

#Berechnung Trägheitsmoment Zylinder grau
schingungsdauer_grauer_zylinder*=(1/5)
schingungsdauer_grauer_zylinder_u=Mittelwert(schingungsdauer_grauer_zylinder)
traeg_zylinder_grau=traegheitsmoment(schingungsdauer_grauer_zylinder_u)
print('Zylinder grau ', traeg_zylinder_grau)

#Berechnung Trägheitsmoment Kugel

schwingungsdauer_kugel*=(1/5)
schwingungsdauer_kugel_u=Mittelwert(schwingungsdauer_kugel)
traeg_kugel=traegheitsmoment(schwingungsdauer_kugel_u)
print('Kugel', traeg_kugel)

#Berechnung Trägheitsmoment Puppe Position 1
schwingungsdauer_puppe_1*=(1/5)
schwingungsdauer_puppe1_u=Mittelwert(schwingungsdauer_puppe_1)
traeg_puppe1_u=traegheitsmoment(schwingungsdauer_puppe1_u)
print('Puppe 1', traeg_puppe1_u)

#Berechnung Trägheitsmoment Puppe Position 2
schwingungsdauer_puppe_2*=(1/5)
schwingungsdauer_puppe2_u=Mittelwert(schwingungsdauer_puppe_2)
traeg_puppe2_u=traegheitsmoment(schwingungsdauer_puppe2_u)
print('Puppe 2', traeg_puppe2_u)







#Plotbereich
plt.xlim(0,0.07)
#plt.ylim(0,52)
x=np.linspace(0,0.28,1000)

plt.plot(abstand[::3]**2,schwingungsdauer_mittel**2,'rx',label='Messwerte')
plt.plot(x**2,m*x**2+b,'b-',label='Lineare Regression')
#plt.plot(abstand[::3]**2,f(abstand[::3]**2,*params_p),'y-',label='Fit')
plt.legend(loc='best')
plt.ylabel(r'$T^2 \ in \ \frac{1}{\mathrm{s}^2}$')
plt.xlabel(r'$r^2 \ in  \ \mathrm{m}^2$')
plt.grid()
#plt.savefig('lineare_regression.pdf')
#plt.show()



#np.savetxt('.txt', np.column_stack([winkelrecht_passiv.n,winkelrecht_passiv.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')