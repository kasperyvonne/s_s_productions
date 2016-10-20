import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

g,m,f,u,u_f=np.genfromtxt('Mittelwert_u.txt', unpack=True)
g,s=np.genfromtxt('Dopplereffekt_pos.txt', unpack=True)

#print(s)
#print(f)


def f(m,u,b):
	return m*u+b
params,covarian=curve_fit(f,u,s)

plt.xlabel('$v$')
plt.ylabel('$\Delta \\nu$')
plt.xlim(0.045,0.23)
plt.ylim(0.8,1.41	)
plt.plot(u,s,'rx',label='Messwert')
plt.plot(u,f(u,*params),'b-',label='Fit')
plt.legend(loc='best')
plt.grid()
plt.title('Geschwindigkeit (positiv) zum berechneten Dopplereffekt (Frequenzmessung1 )')
#plt.show()
plt.savefig('Geschwindigkeit (positiv) zum berechneten Dopplereffekt.pdf')