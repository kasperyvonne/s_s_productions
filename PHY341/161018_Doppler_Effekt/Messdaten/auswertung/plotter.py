import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

u,nu = np.genfromtxt('deltafrequenz_werte_sammlung.txt',unpack=True)
m,m_e,b,b_e=np.genfromtxt('Lingress_deltafrequenz_werte_sammlung.txt',unpack=True)
#print(s)
#print(f)


def f(m,u,b):
	return m*u+b
params_p,covarian=curve_fit(f,u,nu)

def ge(x,m,b):
	return m*u+b

plt.xlabel('$v$')
plt.ylabel('$\Delta \\nu$')
plt.xlim(-0.25,0.25)
#plt.ylim(0.8,1.41	)
x=np.linspace(-0.5,0.5,100)

plt.plot(u,nu,'rx',label='Messwert')
plt.plot(u,f(u,*params_p),'b-',label='Fit')
plt.plot(x,f(x,m,b),'g-',label='Lingress')

plt.legend(loc='best')
plt.grid()
plt.title('Geschwindigkeit zu Schwebung, Frequenzmessung 1')
#plt.show()
plt.savefig('Geschwindigkeit zu allen Deltafrequenzen .pdf')