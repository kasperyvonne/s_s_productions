import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

g,m,f,u,u_f=np.genfromtxt('Mittelwert_u.txt', unpack=True)
t,g_p,s =np.genfromtxt('Mittelwert_schwebung_pos.txt', unpack=True)
t_1,g_n,s =np.genfromtxt('Mittelwert_schwebung_neg.txt', unpack=True)
u_n=-1*u
#print(s)
#print(f)


def f(m,u,b):
	return m*u+b
params_p,covarian=curve_fit(f,u_n,g_p)
def h(m,u,b):
	return m*u+b
params_n, cov=curve_fit(h,u,g_n)

plt.xlabel('$v$')
plt.ylabel('$\Delta \\nu$')
#plt.xlim(0.045,0.23)
#plt.ylim(0.8,1.41	)
plt.plot(u_n,g_p,'rx',label='Messwert')
plt.plot(u_n,f(u_n,*params_p),'b-',label='Fit')
plt.plot(u,g_n,'rx',label='Messwert')
plt.plot(u,h(u,*params_n),'b-',label='Fit')
plt.legend(loc='best')
plt.grid()
plt.title('Geschwindigkeit zu Schwebung, Frequenzmessung 2')
#plt.show()
plt.savefig('Geschwindigkeit zu gemessener Schwebung .pdf')