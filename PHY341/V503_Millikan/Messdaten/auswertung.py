from import_data import *


d = Q_(7.625, 'millimeter')
s = Q_(0.5, 'millimeter')
v = (s/time).to('centimeter/second')
rho_luft = Q_(886, 'kilogram/(meter)**3')
rho_oel = Q_(886, 'kilogram/(meter)**3')
T = temp(resistance.magnitude)

#VISKOSITÄT
eta = viskositaet(T.magnitude)

#TRÖPCHENRADIUS
radius = np.sqrt( (9 * eta * v)/(2*g*(rho_oel))).to('millimeter')

#TRÖPCHENVOLUEM
volume = 4/3 * np.pi * radius**3

#TRÖPFCHENMASSE
mass = (volume * rho_oel)

#ELEKTRISCHE FELDSTÄRKE
E = (voltage/d)

#LADUNG
q_0 = (mass * g / E).to('coulomb')

#KORRIGIERTE LADUNG
p_luft = Q_(1.0132, 'bar')
B = Q_(6.17e-3, 'torr*cm')
q = q_0 * (1 + B/(p_luft * radius))**(-3/2)
print(q)


q_test = np.linspace(1.0e-20, 3e-19, 1000)
F = np.empty((len(q), len(q_test)))
for i in range(0, len(q)):
    F[i] = np.rint(q[i].magnitude/q_test) - q.magnitude[i]/q_test

q_test_best = np.empty(len(q))
for i in range(0, len(q_test_best)):
    q_test_best[i] = q_test[np.argmin(F[i])]
e_exp = Q_(mid(q_test_best), 'coulomb')
r.app(r'e\ua{exp}', e_exp)

#plt.clf()
#plt.plot(q_test, f_q_test, 'rx')
#plt.show()

#N = np.arange(1, len(q) + 1)
#print(q/e_0)
#plt.clf()
#plt.plot(N, q, 'rx')
#plt.show()


r.makeresults()
