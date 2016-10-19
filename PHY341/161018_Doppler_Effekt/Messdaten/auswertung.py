import numpy as np
import uncertainties.unumpy as unp
import math

#Geschwindigkeitsauswertung mit Fehlerrechnun

l=unp.uarray(13e-2,1e-3)

#Hier nicht wichtig
g,u=np.genfromtxt('zeiten.txt',unpack=True)
for a in u:
	u_e=unp.uarray(u,10e-5)

print('Bitte Intervalllänge angeben: ')
m=int(input())
#print(a)

#print(v_e)
g_1=[]
m_u1=[]
m_f=[]
n=0
a=0
f_f=[]
while n<len(u):
	l=sum(u[n:n+m])
	print(l)
	m_u=l/m
	m_u1.append(m_u)
	print('Der Mittelwert für ', g[n], ' lautet: ', m_u)
	g_1.append(g[n])
	hilfsv=0
	n+=m
	while a<n: #hier könnte man das n eingeben
		hilfsv+=(u[a]-m_u)**2
		a+=1
		m_b=np.sqrt(1/(m*(m-1))*hilfsv)
	print('Fehler',m_b)
	print('Dabei ist der Fehler des Mittelwert: ',m_b)
	print('\n')
	m_f.append(m_b)
d=np.array([g_1,m_u1,m_f])


#np.savetxt('Mittelwert_u.txt',d.T,header='Gang Mittelwert Fehler')