import numpy as np
import matplotlib.pyplot as plt
import sys
#Anzahl der Messwerte festlegen
n=[]
m=[]
c=[]
print('Eingabe startet')
u='x'
while 2!=3:
	u=input()
	if u=='end':
		break		
	u_1=float(u)
	n.append(u_1)
	w=float(input())
	m.append(w)
	c.append(float(input()))
	#print(u)
	#rint(w)
#Werte noch übertragen
x=np.array([n,m,c])
x.T

print(x)
print('\n')
np.savetxt('daten.txt',np.column_stack([x]),header='n,m,c')
