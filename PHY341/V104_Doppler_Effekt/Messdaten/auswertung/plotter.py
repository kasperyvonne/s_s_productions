import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

u,nu= np.genfromtxt('schwebung_werte_sammlung.txt',unpack=True)
#print(s)
#print(f)

u*=2
def f(m,u,b):
	return m*u+b
params_p,covarian=curve_fit(f,u,nu)
#
#def ge(x,m,b):
#	return m*x+b
#params_n,covarian=curve_fit(ge,u_n,nu_neg)
#
plt.xlabel('$v$')
plt.ylabel('$\Delta \\nu$')
plt.xlim(-0.5,0.)
#plt.ylim(0.8,1.41	)
x=np.linspace(-0.5,0.5,100)

plt.plot(u,nu,'rx',label='Messwert')
plt.plot(u,f(u,*params_p),'b-',label='Fit')


plt.legend(loc='best')
plt.grid()
plt.title('Geschwindigkeit zu Schwebungsmethode , Messung 2')

plt.savefig('plot_schwebe.pdf')