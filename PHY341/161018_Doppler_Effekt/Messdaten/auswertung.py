import numpy as np
import uncertainties.unumpy as unp
import math

#Geschwindigkeitsauswertung mit Fehlerrechnun

l=unp.uarray(13e-2,1e-3)

g,u=np.genfromtxt('zeiten.txt',unpack=True)
for a in u:
	u_e=unp.uarray(u,10e-5)

#print(a)
v_e=l/u_e
#print(v_e)
g_1=[]
m_u1=[]
m_f=[]
n=0
a=0
print(len(u))
while n<len(u):
	l=sum(u[n:n+5])
	print(l)
	m_u=l/5
	m_u1.append(m_u)
	print('Der Mittelwert für ', g[n])
	print(' lautet ', m_u)
	g_1.append(g[n])
	hilfsv=0
	print(hilfsv)
	n+=5
	while a<n:
		hilfsv+=(u[a]-m_u)**2
		a+=1
		m_b=np.sqrt(1/20*hilfsv)
	
	print('Fehler',m_b)
	m_f.append(m_b)
		
d=np.array([g_1,m_u1,m_f])


#np.savetxt('Mittelwert_u.txt',d.T,header='Gang Mittelwert Fehler')