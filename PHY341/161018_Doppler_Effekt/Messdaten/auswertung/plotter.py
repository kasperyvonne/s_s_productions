import numpy as np
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

g,nu_pos,f_1 = np.genfromtxt('Mittelwert_schwebung_pos.txt',unpack=True)
g_2,nu_neg,f_2=np.genfromtxt('Mittelwert_schwebung_neg.txt',unpack=True)
g_3,m,m_f,u,l_e=np.genfromtxt('Mittelwert_u.txt',unpack=True)
#print(s)
#print(f)
u_n=u*-1
nu_neg*=-1

def f(m,u,b):
	return m*u+b
params_p,covarian=curve_fit(f,u,nu_pos)

def ge(x,m,b):
	return m*x+b
params_n,covarian=curve_fit(ge,u_n,nu_neg)

plt.xlabel('$v$')
plt.ylabel('$\Delta \\nu$')
plt.xlim(-0.25,0.25)
#plt.ylim(0.8,1.41	)
x=np.linspace(-0.5,0.5,100)

plt.plot(u,nu_pos,'rx',label='Messwert')
plt.plot(u,f(u,*params_p),'b-',label='Fit')
plt.plot(u_n,nu_neg,'rx',label='Messwert')
plt.plot(u_n,f(u_n,*params_n),'y-',label='Fit negativ')
#plt.plot(x,f(x,m,b),'g-',label='Lingress')

plt.legend(loc='best')
plt.grid()
plt.title('Geschwindigkeit zu Schwebungsmethode , Messung 2')
#plt.show()
plt.savefig('Geschwindigkeit Schwebungsmethode_betrag_augeloest.pdf')