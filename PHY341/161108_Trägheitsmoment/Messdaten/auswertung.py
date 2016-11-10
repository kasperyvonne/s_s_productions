import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math

#Hier nicht wichtig


abst, wi, kra = np.genfromtxt('winkelrichtgöße.txt',unpack=True)
wi_rad=[]
for w in wi:
	wi_rad.append(math.radians(w))
print(wi_rad)

print('Bitte Intervalllänge angeben: ')


m=int(input())

u=abst

print(u)


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

def D(rad, wink, kraf):
	return (kraf*rad)/wink

winkelricht=[]
while n<len(abst):
	print('Bin dirn')
	winkelricht.append( D(abst[n],wi_rad[n],kra[n]))
	n+=1
print(winkelricht)
winkelricht_mitt=sum(winkelricht)/len(winkelricht)
winkelricht_abweichung_mitt= 1/(np.sqrt(len(winkelricht)))*np.std(winkelricht)

winkelrecht_passiv=ufloat(winkelricht_mitt,winkelricht_abweichung_mitt)

print(winkelrecht_passiv)

print(winkelrecht_passiv.n)

#Berehnung von D dynamisch:


#np.savetxt('Winkelrichtgröße_passiv.txt', np.column_stack([winkelrecht_passiv.n,winkelrecht_passiv.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')