import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Hier nicht wichtig


ab, t = np.genfromtxt('eigenmoment.txt',unpack=True)



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
h_z=0.5*3.49
ab-=h_z
ab*=10e-2

t_hilf=[]
print(ab[::3])
for i in range(len(t))[::3]:
    t_hilf.append(np.mean(t[i:i+3]))
t_mittel=np.array(t_hilf)
print('Zeitlichesmittel ', t_mittel**2)


ab_hilf=[]
print(ab[::3])
for i in range(len(ab))[::3]:
    ab_hilf.append(np.mean(ab[i:i+3]))
ab_mittel=np.array(ab_hilf)
print('Abstands_mittel ', ab_mittel**2)

def f(m,u,b):
	return m*u+b

assert len(ab_mittel)==len(t_mittel)

params_p,covarian=curve_fit(f,ab_mittel**2,t_mittel**2)

plt.xlim(0,4.58)
plt.plot(ab_mittel**2,t_mittel**2,'rx',label='Messwert')
plt.plot(ab_mittel**2,f(ab_mittel**2,*params_p),'b-',label='Fit')
plt.legend(loc='best')

#plt.savefig('Messung_Eigenträgheitsmoment.pdf')
plt.show()



#np.savetxt('Winkelrichtgröße_passiv.txt', np.column_stack([winkelrecht_passiv.n,winkelrecht_passiv.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')