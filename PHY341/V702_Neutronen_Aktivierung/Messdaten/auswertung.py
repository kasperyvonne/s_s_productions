import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l

u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'ohm')
c = Q_(0, 'degC')
c.to('kelvin')
print(c.to('kelvin'))
print(a**2)
print(b**2)
einheitentst=Q_(1*1e-3,'farad')
einheitentst_2=Q_(1,'ohm')
print(einheitentst)
print(1/(einheitentst*einheitentst_2).to('second'))


#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

#Standartabweichung und Mittelwert

def mittel_und_abweichung(messreihe):
	messreihe_einheit=messreihe.units
	mittelwert=sum(messreihe)/len(messreihe)
	abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe)))*np.std(messreihe)
	mittel_und_abweichung=Q_(unp.uarray(mittelwert,abweichung_des_mittelwertes),messreihe_einheit)
	return mittel_und_abweichung

#Standartabweichung und Mittelwert für Messreihe mit Intervallen
def mittel_und_abweichung_intervall(messreihe,intervall_laenge):
	messreihe_einheit=messreihe.units
	mittelwert_abweichung_liste=[]
	for i in range(len(messreihe))[::intervall_laenge]:
		mittelwert=sum(messreihe[i:i+intervall_laenge])/len(messreihe[i:i+intervall_laenge])
		abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe[i:i+intervall_laenge])))*np.std(messreihe[i:i+intervall_laenge])
		mittelwert_abweichung_liste.append(ufloat(mittelwert.magnitude,abweichung_des_mittelwertes.magnitude))
	mittelwert_abweichung_u=Q_(unp.uarray(unp.nominal_values(mittelwert_abweichung_liste),unp.std_devs(mittelwert_abweichung_liste)),messreihe_einheit)
	return mittelwert_abweichung_u


#Lineare regression

def linregress(x, y):
    assert len(x) == len(y)

    x, y = np.array(x), np.array(y)

    N = len(y)
    Delta = N * np.sum(x**2) - (np.sum(x))**2

    # A ist die Steigung, B der y-Achsenabschnitt
    A = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)

    return A, A_error, B, B_error


#Angepasstes Programm

def fehler_log(anzahl):
	fehler=np.sqrt(anzahl)
	obere_schranke=np.log(anzahl+fehler)-np.log(anzahl)
	untere_schranke=np.log(anzahl-fehler)-np.log(anzahl)
	untere_schranke*=(-1)
	anzahl_fehler=[untere_schranke,obere_schranke]
	anzahl_fehlerarray=np.array(anzahl_fehler)
	return anzahl_fehlerarray

nullrate=167
nullrate*=(240)/(900)



##Isotop mit einem eifachen Zerfall

zeit,anzahl= np.genfromtxt('zaehlrate_indium.txt',unpack=True)
zeit*=240
anzahl-=nullrate
anzahl_fehler=np.sqrt(anzahl)
anzahlu=unp.uarray(anzahl,anzahl_fehler)


fehlelog_indium=fehler_log(anzahl)

l.Latexdocument('messwerte_indium.tex').tabular([zeit, anzahl, anzahl_fehler, fehlelog_indium[1], fehlelog_indium[0] ],
'Jo', [0, 1, 1,2,2] ,
caption = 'Gemessene Anzahl an Zerfällen bei Indium', label = 'tab: indium_messwerte')

#latex.Latexdocument('messwerte_indium.tex').tabular([zeit, anzahl, anzhal_fehler, fehler_log[1], fehler_log[0] ],
#'{$t$ in $\si{\secod}$} & {Anzahl $N$} & {$\sigma\ua{N}$} & {$\ln{N+\sigma\ua{N}}-\ln{N}$} & {$\ln{N-\sigma\ua{N}}-\ln{N}$}', [1, 1, 1,1 ,1 ] ,
#caption = 'Gemessene Anzahl an Zerfällen bei Indium', label = 'tab: indium_messwerte')


def g(x,m,b):
	return m*x+b

params_indium, cov_indium = curve_fit(g, zeit, np.log(anzahl) )
indium_errors = np.sqrt(np.diag(cov_indium))
indium_u =unp.uarray(params_indium,indium_errors)

print('Steigung', indium_u[0])
print('Achsenabschnitt', indium_u[1])
print('\n')






##Isomer Rhodoium
nullrate_rhodium=218
nullrate_rhodium*=(15)/(900)
zeit_rhodium, anzahl_rhodium=np.genfromtxt('zaehlreite_rhodium.txt', unpack=True)
zeit_rhodium*=15
anzahl_rhodium-=nullrate_rhodium
print(zeit_rhodium)

anzahl_rhodium_fehler=np.sqrt(anzahl_rhodium)
fehlerlog_rhodium=fehler_log(anzahl_rhodium)


t_sternchen=500
params_rho_lang, cov_rho_lang = curve_fit(g, zeit_rhodium[33:-1], np.log(anzahl_rhodium[33:-1]) )
rho_lang_errors = np.sqrt(np.diag(cov_rho_lang))
rho_lang_u =unp.uarray(params_rho_lang,rho_lang_errors)

print('Steigung rho lang', rho_lang_u[0])
print('Achsenabschnitt', rho_lang_u[1])
print('\n')

params_rho_gesamt, cov_rho_gesamt = curve_fit(g, zeit_rhodium, np.log(anzahl_rhodium) )
rho_gesamt_errors = np.sqrt(np.diag(cov_rho_gesamt))
rho_gesamt_u =unp.uarray(params_rho_gesamt,rho_gesamt_errors)

print('Steigung rho gesamt', rho_gesamt_u[0])
print('Achsenabschnitt', rho_gesamt_u[1])
print('\n')





anzahl_rhodium_kurz_gesamt=anzahl_rhodium[0:10]
anzahl_rhodium_lang_zumzeitpunkt_kurz=[]
for t in zeit_rhodium[0:10]:
	anzahl_rhodium_lang_zumzeitpunkt_kurz.append(g(t,params_rho_gesamt[0],params_rho_gesamt[1]))
anzahl_rhodium_kurz=np.log(anzahl_rhodium_kurz_gesamt-anzahl_rhodium_lang_zumzeitpunkt_kurz)

params_rho_kurz, cov_rho_kurz = curve_fit(g, zeit_rhodium[0:10], anzahl_rhodium_kurz )
rho_kurz_errors = np.sqrt(np.diag(cov_rho_kurz))
rho_kurz_u =unp.uarray(params_rho_kurz,rho_kurz_errors)

print('Steigung rho kurz', rho_kurz_u[0])
print('Achsenabschnitt', rho_kurz_u[1])
print('\n')



rho_lang=g(t_sternchen,params_rho_lang[0],params_rho_lang[1])


print('RHO lang nach sternchen', rho_lang)
print('rho kurz nach sternchen', g(t_sternchen,params_rho_kurz[0],params_rho_kurz[1]))
print('\n')


l.Latexdocument('messwerte_rhodium.tex').tabular([zeit_rhodium, anzahl_rhodium, anzahl_rhodium_fehler, fehlerlog_rhodium[1], fehlerlog_rhodium[0] ],
'Jo', [2, 2, 2,2,2] ,
caption = 'Gemessene Anzahl an Zerfällen bei Rhodium', label = 'tab: rhodium_messwerte')


#Plotbereich
yerr=np.zeros(15)
##Plot indium nicht loga
plt.clf()
#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()

plt.plot(zeit,anzahl,'rx',label=r'Gemessene Zerfälle')

plt.errorbar( zeit, anzahl + yerr,  yerr=anzahl_fehler, fmt='x')

plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Zeit in s')
plt.ylabel(r'Gemessene Zerfälle')
#plt.show()
#plt.savefig('.pdf')

##Plot indium loga
t_indium=np.linspace(0,zeit[-1]+1000,1000)
plt.clf()
plt.errorbar( zeit, np.log(anzahl), yerr=fehlelog_indium, fmt='x',label=r'Gemessene Zerfälle')
plt.plot(t_indium,g(t_indium, params_indium[0],params_indium[1]), 'r-', label=r'Regeressionsgerade')
plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Zeit in s')
plt.ylabel(r'Gemesene Zerfälle, logarithmiert')
#plt.show()
plt.xlim(0,4000)
plt.savefig('logarithmiert_indium.pdf')

##Plot rhodium

plt.clf()
#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()
plt.plot(zeit_rhodium,anzahl_rhodium,'rx',label=r'Gemessene Zerfälle')
plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Zeit in s')
plt.ylabel(r'Gemessene Zerfälle')
#plt.show()
#plt.savefig('.pdf')


##plot rhodium lang und kurz
plt.clf()
t_1=np.linspace(0,zeit_rhodium[8],1000)
t_2=np.linspace(490,800,1000)
t_g=np.linspace(0,800,1000)
y = np.linspace(0, 10, 10)
plt.errorbar( zeit_rhodium, np.log(anzahl_rhodium), yerr=fehlerlog_rhodium, fmt='x',label=r'Gemessene Zerfälle')
plt.plot(t_g,g(t_g,params_rho_gesamt[0],params_rho_gesamt[1]),'r-',label=r'Zerfallsfit für die gesamte Messung')
plt.plot(t_2,g(t_2,params_rho_lang[0],params_rho_lang[1]),'y-',label=r'Zerfallsfit für Ra*')
plt.plot(t_1,g(t_1,params_rho_kurz[0],params_rho_kurz[1]),'g-',label=r'Zerfallsfit für Ra')
plt.plot([t_sternchen for i in y], y, 'c-',label=r't*=500s')
plt.grid()
plt.legend(loc='best')
plt.xlabel(r'Zeit in s')
plt.ylabel(r'Gemessene Zerfälle, logarithmiert')
#plt.show()
plt.savefig('ra.pdf')
