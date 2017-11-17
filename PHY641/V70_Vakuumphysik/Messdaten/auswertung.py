import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
from scipy.stats import sem
import latex as l
#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS
a = [1.1, 2.2, 3.3]
b = [0.1, 0.2, 0.3]
c = [3, 4, 5]
d = [6, 7, 8] #EXAMPLE ARRAYS

f = unp.uarray(a, b) #EXAMPLE UARRAY

l.Latexdocument('latex_example.tex').tabular(
data = [c, d, f], #Data incl. unpuarray
header = ['\Delta Q / \giga\elementarycharge', 'T_1 / \micro\second', 'T_1 / \micro\second'],
places = [1, 1, (1.1, 1.1)],
caption = 'testcaption.',
label = 'testlabel')
##############################################


## Funktionen ##

def mittelwert_zeit(t_1,t_2,t_3,t_4,t_5): ## Funktion um die Zeitmittelwerte der Messreihe zu bestimmen.
    n=len(t_1)
    mittel=[]
    fehler=[]

    for i in range(n):
        zeit= [ t_1[i], t_2[i], t_3[i], t_4[i], t_5[i] ]
        mittel.append(np.mean(zeit))
        fehler.append(sem(zeit))

    return unp.uarray(mittel,fehler)

def mittelwert_zeit_leck(t_1,t_2,t_3): ## Funktion um die Zeitmittelwerte der Messreihe zu bestimmen etwas unnötig.
    n=len(t_1)
    mittel=[]
    fehler=[]

    for i in range(n):
        zeit= [ t_1[i], t_2[i], t_3[i] ]
        mittel.append(np.mean(zeit))
        fehler.append(sem(zeit))

    return unp.uarray(mittel,fehler)


def g(m,x,b):
    return m*x+b





###  Drehschieber
#Konstanten
enddruck=ufloat(1e-2, 1e-2*0.2)


p_druck, t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5 = np.genfromtxt('./messdaten/drehshieber/drehschieber_druck.txt', unpack=True)

preasure_druck=unp.uarray(p_druck,p_druck*0.2)

zeiten_dreh_druck_gemittelt=mittelwert_zeit( t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5)

lograritmierter_druck=unp.log( (preasure_druck-enddruck )/ (preasure_druck[0]-enddruck) )


## Druckkurve


# Geradenfir Bereich 1
print('\n \n \n')
parms_druck_schiber_1, cov_druck_schieber_1 = curve_fit(g,noms(zeiten_dreh_druck_gemittelt[1:4]), noms(lograritmierter_druck[1:4]) )
error_druck_schieber_1= np.sqrt(np.diag(cov_druck_schieber_1))
m_u_druck_schieber_1=ufloat(parms_druck_schiber_1[0],error_druck_schieber_1[0])
b_u_druck_schieber_1=ufloat(parms_druck_schiber_1[1],error_druck_schieber_1[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 1', m_u_druck_schieber_1)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 1', b_u_druck_schieber_1)
print('\n')
# Geradenfit Bereich 2
parms_druck_schiber_2, cov_druck_schieber_2 = curve_fit(g,noms(zeiten_dreh_druck_gemittelt[4:12]), noms(lograritmierter_druck[4:12]) )
error_druck_schieber_2= np.sqrt(np.diag(cov_druck_schieber_2))
m_u_druck_schieber_2=ufloat(parms_druck_schiber_2[0],error_druck_schieber_2[0])
b_u_druck_schieber_2=ufloat(parms_druck_schiber_2[1],error_druck_schieber_2[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 2', m_u_druck_schieber_2)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 2', b_u_druck_schieber_2)
print(' \n')
# Geradenfit Bereich 3
parms_druck_schiber_3, cov_druck_schieber_3 = curve_fit(g,noms(zeiten_dreh_druck_gemittelt[12:15]), noms(lograritmierter_druck[12:15]) )
error_druck_schieber_3= np.sqrt(np.diag(cov_druck_schieber_3))
m_u_druck_schieber_3=ufloat(parms_druck_schiber_3[0],error_druck_schieber_3[0])
b_u_druck_schieber_3=ufloat(parms_druck_schiber_3[1],error_druck_schieber_3[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 3', m_u_druck_schieber_3)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 3', b_u_druck_schieber_3)
print('\n \n \n')

# Plot der Druckkurve mit den Fitgeraden
t_1=np.linspace(noms(zeiten_dreh_druck_gemittelt[1])-4,noms(zeiten_dreh_druck_gemittelt[3]+4),1000)
t_2=np.linspace(noms(zeiten_dreh_druck_gemittelt[4])-4,noms(zeiten_dreh_druck_gemittelt[11]+4),1000)
t_3=np.linspace(noms(zeiten_dreh_druck_gemittelt[12])-4,noms(zeiten_dreh_druck_gemittelt[14]+4),1000)


plt.grid()
plt.errorbar(noms(zeiten_dreh_druck_gemittelt),noms(lograritmierter_druck), xerr=stds(zeiten_dreh_druck_gemittelt), yerr=stds(lograritmierter_druck),fmt='.',label='Messwerte')
plt.plot(t_1, noms(m_u_druck_schieber_1)* t_1+ noms(b_u_druck_schieber_1), label='Regressionsgerade 1')
plt.plot(t_2, noms(m_u_druck_schieber_2)* t_2+ noms(b_u_druck_schieber_2), label='Regressionsgerade 2')
plt.plot(t_3, noms(m_u_druck_schieber_3)* t_3+ noms(b_u_druck_schieber_3), label='Regressionsgerade 3')
plt.xlabel(r'$ t \, / \, s $')
plt.ylabel(r'$ \ln\left( \frac{ p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}} } \right) $')
plt.legend()
plt.savefig('./plots/dreh/druckplot_drehschieber.pdf')

# Tabelle für die Druckkurve mit den Fitgeraden
l.Latexdocument('./table/dreh/druck_messdaten.tex').tabular(
data = [p_druck, lograritmierter_druck,t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5, zeiten_dreh_druck_gemittelt], #Data incl. unpuarray
header = ['p(t) / \milli\bar', '\ln( \frac{p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}}}', 't_1 / \second', 't_2 / \second',  't_3 / \second',  't_4 / \second',  't_5 / \second', '\overline{t} / \second'],
places = [1, (1.1, 1.1), 1, 1, 1, 1, 1, (1.1, 1.1)],
caption = 'Für die Bestimmung des Saugvermögens $S$ der Drehschieberpumpe gemessene Drücke. Die Messung wurde bei Raumtemperatur durchgeführt. Es ist $p_{\mathrm{g}}=\SI{1e-2\pm 2e-4}{\milli\bar}$ der Enddruck und  $p_{\mathrm{g}}=\SI{1e3}{\milli\bar}$',
label = 'druck_dreh')

## Leckkurve

#Leckkurve für den druck 1mbar

p_dreh_leck_1, t_1_dreh_leck_1, t_2_dreh_leck_2, t_1_dreh_leck_3 = np.genfromtxt('./messdaten/drehshieber/drehschieber_leck_1.txt',unpack=True)

t_gemittel_dreh_leck_1=mittelwert_zeit_leck(t_1_dreh_leck_1,t_2_dreh_leck_2,t_1_dreh_leck_3)


plt.clf()
plt.plot(noms(t_gemittel_dreh_leck_1),p_dreh_leck_1,'.')
#plt.show()

def auswertung_leck(p, t_1, t_2, t_3,name):
    t_gemittelt=mittelwert_zeit_leck(t_1,t_2,t_3)
    messwerte={}

    parms, cov = curve_fit(g,noms(t_gemittelt),p )
    error= np.sqrt(np.diag(cov))
    m_u=ufloat(parms[0],error[0])
    b_u=ufloat(parms[1],error[1])
    print(' Steigung der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',m_u )
    print(' y-Achsen der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',b_u )
    messwerte['Steigung']=m_u
    messwerte['Achsenabschnitt']=b_u

    laufvariabele=np.linspace(noms(t_gemittelt[0])-1, noms(t_gemittelt[-1])+1,10000)
    plt.grid()
    plt.errorbar(noms(t_gemittelt),p, xerr=stds(t_gemittelt), yerr=None,fmt='.',label='Messwerte')
    plt.plot(laufvariabele, noms(m_u)* laufvariabele+ noms(b_u), label='Regressionsgerade')
    plt.xlabel(r'$ t \, / \, s $')
    plt.ylabel(r'$ p \, / \, mbar $')
    plt.legend()
    plt.savefig('./plots/dreh/leckrate_' + name +'_'+ str(p[0]) + '.pdf')

    l.Latexdocument('./table/'+name+'_tabelle.tex').tabular(
    data = [p, t_1,t_2,t_3, t_gemittelt], #Data incl. unpuarray
    header = ['p / \milli\bar', 't_1 / \second', 't_2 / \second','t_3 / \second', '\overline{t} / \second'],
    places = [1, 1, 1, 1, (1.1, 1.1)],
    caption = 'Gemessene Drücke bei der Leckkratenmethode für die Drehschieberpumpe mit $p_{\mathrm{l}}=' +str(p[0]) +'$. Messung bei Raumtemperatur.',
    label = 'leck_' + name + '_leck_'+ str(p[0]) + '.pdf' )

    return messwerte

auswertung_leck(p_dreh_leck_1, t_1_dreh_leck_1, t_2_dreh_leck_2, t_1_dreh_leck_3,'dreh')












###  Turbo

## Druckkurve

## Leckkurve
