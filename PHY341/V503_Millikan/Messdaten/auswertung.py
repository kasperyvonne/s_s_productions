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
#q = (q_0 * (1 + B/(p_luft * radius))**(-3/2)).to('coulomb')

r_korr = np.sqrt((B/(2 * p_luft))**2  +  ((9 * eta*v)/(2 * g * rho_oel))) - B/(2 * p_luft)
volume = 4/3 * np.pi * r_korr**3

#TRÖPFCHENMASSE
mass = (volume * rho_oel)

#ELEKTRISCHE FELDSTÄRKE
E = (voltage/d)

q =(mass * g / E).to('coulomb')




#sp StartPunkt der Beachtung der Messwerte
sp = 5
q_sort = np.sort(q)[sp:]

# dist, Liste mit allen möglichen Differenezen
dist = list()
for i in range(1, len(q_sort)):
    for j in range(i+1, len(q_sort)):
        dist.append(abs(q_sort[i] - q_sort[j]))
#dist = np.sort(dist)
N = np.arange(1, len(dist) + 1)
#plt.clf()
#plt.plot(N, dist, 'rx')
#plt.show()

dist = np.sort(dist)

dist_min = dist[1.5 * sp/25 * len(dist)]
print(dist_min)
range_d = 1e-19

q_e = dist_min
r.app(r'q\ua{e}', Q_(q_e, 'coulomb'))
#1e-20, 3e-19
q_test = np.linspace(dist_min - range_d, dist_min + range_d, 1000)


F = np.empty((len(q_sort), len(q_test)))
for i in range(0, len(q_sort)):
    F[i] = abs(np.rint(q_sort[i]/q_test) - q_sort[i]/q_test)

q_test_best = np.empty(len(q_sort))
for i in range(0, len(q_test_best)):
    q_test_best[i] = q_test[np.argmin(F[i])]

N = np.arange(1, len(q[sp:]) + 1)

ticks = list()
for i in N:
    ticks.append('$q_{' + str(i) + '}$')
plt.clf()
plt.axhline(y = e_0.magnitude, ls='-', color='r', linewidth= 1, label= 'Elementarladung $e_0$')
plt.plot(N, q_test_best, 'bx', label='Bestwerte für die Elementarladung $e_i$')
plt.ylabel('$e_i / 10^-19 C$')
plt.grid()
plt.legend(loc = 'best')
plt.xticks(N, ticks)
plt.savefig('scattering.pdf')




e_exp = abs(Q_(mid(q_test_best), 'coulomb'))
r.app(r'e\ua{exp}', e_exp)
r.app(r'proz', e_exp/e_0 - 1)


r.makeresults()
