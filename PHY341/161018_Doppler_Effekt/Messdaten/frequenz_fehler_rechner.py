import numpy as np

g,u=np.genfromtxt('schwebung_v_neg.txt',unpack=True)
#print (u)
du=[]
for i in u:
	du.append(i*10e-5)

np.savetxt('schwebung_v_neg_mit_fehler.txt',np.column_stack([g,u,du]),header='Gang Frequenz Fehler')