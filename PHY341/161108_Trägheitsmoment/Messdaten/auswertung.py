import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Hier nicht wichtig


abstand, schwingungsdauer = np.genfromtxt('eigenmoment.txt',unpack=True)



print('Bitte Intervalllänge angeben: ')


m=int(input())



g_1= [] 
m_u1= []
m_f=[] 
f_f=[] 
v_m=[]
v_f=[]
dopp=[]
dopp_f=[]
n =0
a=0
b=13e-2
b_f=1e-3

#while n<len(u):
#	l=sum(u[n:n+m])
#	m_u=l/m
#	m_u1.append(m_u)
#	print('Der Mittelwert lautet: ', m_u)
#	hilfsv=0
#	n+=m
#	while a<n:
#		hilfsv+=(u[a]-m_u)**2
#		a+=1
#	m_b=np.sqrt(1/(m*(m-1))*hilfsv)
#	m_f.append(m_b)	
#	m_un=unp.uarray(m_u,m_b)
#	print('Dabei ist der Fehler des Mittelwert: ',m_b)
#	print('\n')
	
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
schwingungsdauer*=0.2
schwingungsdauer_hilf=[]

print(abstand[::3])
for i in range(len(schwingungsdauer))[::3]:
    schwingungsdauer_hilf.append(np.mean(schwingungsdauer[i:i+3]))
schwingungsdauer_mittel=np.array(schwingungsdauer_hilf)
print('Zeitlichesmittel ', schwingungsdauer_mittel**2)


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


plt.xlim(0,6.4)
plt.ylim(0,52)
x=np.linspace(0,6.5,1000)

plt.plot(abstand[::3]**2,schwingungsdauer_mittel**2,'rx',label='Messwerte')
plt.plot(x**2,m*x**2+b,'b-',label='Lineare Regression')
#plt.plot(ab[::3]**2,f(ab[::3]**2,*params_p),'b-',label='Fit')
plt.legend(loc='best')
plt.ylabel(r'$T^2 \ in \ \frac{1}{\mathrm{s}^2}$')
plt.xlabel(r'$r^2 \ in  \ \mathrm{m}^2$')
plt.grid()
plt.savefig('lineare_regression.pdf')
#plt.show()



#np.savetxt('Winkelrichtgröße_passiv.txt', np.column_stack([winkelrecht_passiv.n,winkelrecht_passiv.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')