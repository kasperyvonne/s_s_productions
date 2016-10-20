import numpy as np
import uncertainties.unumpy as unp
import math

#Geschwindigkeitsauswertung mit Fehlerrechnun

l_1=unp.uarray(13e-2,1e-3)

#Hier nicht wichtig
g,u=np.genfromtxt('frequenz_v_neg.txt',unpack=True)
n_0=np.genfromtxt('ruhefrequenz.txt',unpack=True)
for a in u:
	u_e=unp.uarray(u,10e-5)

print('Bitte Intervalllänge angeben: ')
m=int(input())
n_0m=sum(n_0)/len(n_0)
n_0f=np.std(n_0,ddof=1)/(np.sqrt(len(n_0)))
n_s=(n_0m,n_0f)

#for N_0 in n_0:
#	n_0f+=np.sqrt(1/m*(m-1)*(N_0-n_0m))
#
print('Mittelwert Ruhefrequenz ', n_0m)
print('Standartabweihung des Ruhefrequenz Mittelswertes ', n_0f)

#print(v_e)
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
while n<len(u):
	l=sum(u[n:n+m])
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
	dopp.append(n_0m-m_u)
	print('Fehler',m_b)
	print('Dabei ist der Fehler des Mittelwert: ',m_b)
	print('\n')
	#v=b/m_u
	##print('Geschwindigkeit', v)
	#v_m.append(v)
	#v_f.append(np.sqrt((m_b/m_u)**2+(b_f/b)))
	#m_f.append(m_b)


d=np.array([g_1,m_u1,m_f,v_m,v_f])

#np.savetxt('Mittelwert_u.txt',d.T,header='Gang Mittelwert Fehler Geschwindgkeit Fehler_Geschwindigkeit')
np.savetxt('ruhefrequenz_mittelwert.txt',n_0s,header='Mittelwert Abweichung')