from load_data import *
####FUNKTIONEN
def bessel(g, b):
    d = g - b
    e = g + b
    return (e**2 - d**2)/(4 * e)

def lin(x, m, b):
    return m * x + b

def linfit(x, y):
    params_raw, cov = curve_fit(lin, x, y)
    return correlated_values(params_raw, cov)


####MITTELWERT BRENNWEITE 1
method_1_f = (1 / (1 / method_1_g + 1 / method_1_b)).mean()
r.app(r'f\ua{1}', Q_(method_1_f, 'centimeter'))



######################


######BESSELMETHODE
method_bessel_f = bessel(bessel_g, bessel_b).mean()
r.app(r'f\ua{2}', Q_(method_bessel_f, 'centimeter') )

#######CHROMATISCHE ABBERATION
brennweite_blau = bessel(blau_g, blau_b).mean()
r.app(r'f\ua{b}', Q_(brennweite_blau, 'centimeter') )

brennweite_rot = bessel(rot_g, rot_b).mean()
r.app(r'f\ua{r}', Q_(brennweite_rot, 'centimeter') )


#####ABBE
x = np.linspace(1.4, 3, 1000)
plt.errorbar(unp.nominal_values(1 + 1/abbe_V), unp.nominal_values(abbe_g),
xerr = unp.std_devs(1 + 1/abbe_V), yerr = unp.std_devs(abbe_g), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
params_g = linfit(unp.nominal_values(1 + 1/abbe_V), unp.nominal_values(abbe_g))
m_g = params_g[0]
r.app(r'f\ua{a, 1}', Q_(m_g, 'centimeter'))
b_g = params_g[1]
r.app('h', Q_(b_g, 'centimeter'))
plt.plot(x, lin(x, m_g.n, b_g.n), 'r-', label='Lineare Regression')
plt.grid()
plt.xlabel(r'$\left(1 + \frac{1}{V}\right)$')
plt.ylabel(r'$g$/cm')
plt.xlim(1.6, 2.9)
plt.legend(loc='best')
plt.savefig('plots/abbe_plot_g.pdf')
plt.clf()

plt.errorbar(unp.nominal_values(1 + abbe_V), unp.nominal_values(abbe_b),
xerr = unp.std_devs(1 + abbe_V), yerr = unp.std_devs(abbe_b), fmt='bx',
ecolor = 'b', elinewidth = 1, capsize = 2, label='Messwerte')
params_b = linfit(unp.nominal_values(1 + abbe_V), unp.nominal_values(abbe_b))
m_b = params_b[0]
r.app(r'f\ua{a, 2}', Q_(m_b, 'centimeter'))
b_b = params_b[1]
r.app('h-', Q_(b_b, 'centimeter'))
plt.plot(x, lin(x, m_b.n, b_b.n), 'r-', label='Lineare Regression')
plt.grid()
plt.xlabel(r'$\left(1 + V\right)$')
plt.ylabel(r'$b$/cm ')
plt.xlim(1.5, 2.65)
plt.legend(loc='best')
plt.savefig('plots/abbe_plot_b.pdf')
plt.clf()



######WASSERLINSE
wasser_f = (1 / (1 / wasser_g + 1 / wasser_b) ).mean()
r.app(r'f\ua{w}', Q_(wasser_f, 'centimeter'))
plt.plot([wasser_g_raw[0], 0], [0, wasser_b_raw[0]], 'k-', label= 'Verbindungslinien der Wertepaare $(g_i, b_i)$')
for i in range(1, len(wasser_g_raw)):
    plt.plot([wasser_g_raw[i], 0], [0,wasser_b_raw[i]], 'k-')
plt.grid()
plt.xlim(0, 36)
plt.ylim(0, 49)
plt.axvline(x = 14.1, ls='-', color='r', label = 'Abgelesene Brennweite')
plt.axvline(x = wasser_f.n, ls='--', color='b', label = 'Berechnete Brennweite')
plt.legend(loc='best')
plt.savefig('plots/wasserlinse.pdf')
plt.clf()


######PLOTS
######METHODE 1

plt.plot([noms(method_1_g[::-1])[0], 0], [0,noms(method_1_b)[0]], 'k-', label= 'Verbindungslinien der Wertepaare $(g_i, b_i)$')
for i in range(1, 10):
    plt.plot([noms(method_1_g[::-1])[i], 0], [0,noms(method_1_b)[i]], 'k-')
plt.grid()
plt.axvline(x = 15, ls='-', color='r', label = 'Abgelesene Brennweite')
plt.axvline(x = method_1_f.n, ls='--', color='b', label = 'Berechneter Mittelwert aus Linsenformel')
plt.xlim(0, 38)
plt.ylim(0, 23)
plt.legend(loc='best')
plt.savefig('plots/methode_1.pdf')
plt.clf()

#for i in range(0, len(test_g)):
#    plt.plot([test_g[i], 0], [0,test_b[i]], 'k-')
#plt.show()
r.makeresults()
