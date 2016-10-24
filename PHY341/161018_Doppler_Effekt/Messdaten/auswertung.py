import numpy as np
import uncertainties.unumpy as unp
import math

#Geschwindigkeitsauswertung mit Fehlerrechnun

l_1=unp.uarray(13e-2,1e-3)

#Hier nicht wichtig
g,u=np.genfromtxt('zeiten.txt',unpack=True)
well=np.genfromtxt('wellenlaenge.txt', unpack=True)
n_0=np.genfromtxt('ruhefrequenz.txt',unpack=True)


for a in u:
	u_e=unp.uarray(u,10e-5)

well*=2e-3
well_m=sum(well)/len(well)
wellf=np.std(well,ddof=1)/(np.sqrt(len(well)))
wellu=unp.uarray(well_m,wellf)
inwell=1/wellu

print('Bitte Intervalllänge angeben: ')


m=int(input())

n_0m=sum(n_0)/len(n_0)
print('Mittelwert f_0: ', n_0m)
n_0f=np.std(n_0,ddof=1)/(np.sqrt(len(n_0)))
n_s=(n_0m,n_0f)

c_m=n_0m*well_m
c_f=np.sqrt((wellf/well_m)**2+(n_0f/n_0m)**2)




n_0un=unp.uarray(n_0m,n_0f)
n_0u=np.array( [n_0m, n_0f])

print(n_0un)
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
	m_f.append(m_b)	
	m_un=unp.uarray(m_u,m_b)
	dopp.append(m_u-n_0m)
	dopp_f.append(unp.std_devs(n_0un-m_un))
	print('Fehler vom Delta', (n_0un-m_un))
	print('Fehler',m_b)
	print('Dabei ist der Fehler des Mittelwert: ',m_b)
	print('\n')
	v=b/m_u
	print('Geschwindigkeit', v)
	v_m.append(v)
	v_f.append(np.sqrt((m_b/m_u)**2+(b_f/b)**2))
	
dopp_un=unp.uarray(dopp,dopp_f)
dopp_u=np.array([dopp,dopp_f])

print(inwell)
print(g)
print(v)
print(v_f)
d=np.array([g_1,v_m,v_f])
#np.savetxt('Wellenaenge_Mittelwert.txt',np.column_stack([well_m, wellf]),header='Mittelwert Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')