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

def saufkraft_leck(p_g,V,m):
    return p_0/V *m

def saukraft_druck(m,V):
    return -m*V

def volu(r,h):
    return np.pi*r**2*h

### Volumenbestimmung alle einheiten in m

## Volumen von beiden
print('-----------------------------------------------------------------')
print(' Volumen beider Pumpen \n-----------------------------------------------------------------\n')
#Volumen Tank
r_t_1=ufloat(7.6e-2-3e-3,(3/2)*1e-3)
h_t_1=ufloat(49.5e-2, 0.5e-3)
v_t_1=volu(r_t_1,h_t_1)

r_t_2=ufloat((1/2*38.15)*1e-3 - 3e-3,(3/2)*1e-3)
h_t_2=ufloat(8.5e-2,(0.5)*1e-3)
v_t_2=volu(r_t_2,h_t_2)

r_t_3=ufloat((1/2*2.9)*1e-2-3e-3, (3/2)*1e-3)
h_t_3=ufloat(6.1e-2, 0.5*1e-3)
v_t_3=volu(r_t_3,h_t_3)

v_t=v_t_1+v_t_2+v_t_3
v_t*=1e3 # Umrechnung in Liter
print('Volumen Tank: ',v_t,'\n')

# Volumen Verbindung Tank
r_vt=ufloat((1/2*61.9)*1e-3,0.5*1e-3)
h_vt=ufloat(60.2e-3, (1/2)*1e-3)
v_vt=volu(r_vt,h_vt)*1e3 # Umrechnung in liter
print('Volumen Verbindung zum Tank: ',v_vt,'\n')

#Volumen 2 Verbindung Tank

r_v2t=ufloat((1/2*61.9)*1e-3,0.5*1e-3)
h_v2t=ufloat(46.4e-3, (1/2)*1e-3)
v_v2t=volu(r_v2t,h_v2t)*1e3 # Umrechnung in liter
print('Volumen Verbindung zum Tank 2: ', v_v2t,'\n')

#Volumen Kugelventil

r_kv=ufloat((1/2*11.7)*1e-3,0.5*1e-3)
h_kv=ufloat(74.6e-3, (1/2)*1e-3)
v_kv=volu(r_kv,h_kv)*1e3 # Umrechnung in liter
print('Volumen eines Kugelventils: ', v_kv,'\n')

#Volumen Kreuzung beim Tank

r_kbt_1=ufloat((1/2*39.2)*1e-3,0.5*1e-3)
h_kbt_1=ufloat(130.2e-3, 0.5*1e-3)
v_kbt_1=volu(r_kbt_1,h_kbt_1)

r_kbt_2=ufloat((1/2*15.6)*1e-3,0.5*1e-3)
h_kbt_2=ufloat((2*27.6)*1e-3,0.5*1e-3)
v_kbt_2=volu(r_kbt_2,h_kbt_2)

v_kbt=(v_kbt_1+v_kbt_2)*1e3 #Umrechnung in Liter
print('Volumen der Kreuzung beim Tank: ', v_kbt, '\n')

#Volumen Glühkathode
r_gk_1=ufloat((1/2*39.2)*1e-3,0.5*1e-3)
h_gk_1=ufloat(47.9e-3,0.5*1e-3)
v_gk_1=volu(r_gk_1,h_gk_1)

r_gk_2=ufloat((1/2*39)*1e-3,0.5*1e-3)
h_gk_2=ufloat(164.8e-3,0.5*1e-3)
v_gk_2=volu(r_gk_2,h_gk_2)
v_gk=(v_gk_1+v_gk_2)*1e3 #Umrechnung in Liter

print('Volumen der Glühkathode: ', v_gk, '\n')

# Volumen Schlauch zur zweiten Kreuzung

r_sk=ufloat((1/2*17)*1e-3,(0.5)*1e-3)
h_sk=ufloat(43e-2,0.5*1e-3)
v_sk=volu(r_sk,h_sk)*1e3

print('Volumen Schlauch zur Kreuzung: ', v_sk, '\n')

# Volumen Kreuzung zur Drehschieberpumpe

r_dk=ufloat((1/2*12.3)*1e-3, (0.5)*1e-3)
h_dk=ufloat( 2*80.3e-3, (0.5)*1e-3)
v_dk=volu(r_dk,h_sk)*1e3 #Umrechnung in Liter

print('Volumen des Kreuzungsstück zur Drehschieberpumpe: ', v_dk, '\n')

# Volumen des T-Stücks zum Messgeräts

r_tm=ufloat( (1/2*12.5)*1e-3, (0.5)*1e-3)
h_tm=ufloat( 110e-3, (0.5)*1e-3)
v_tm=volu(r_tm, h_tm)*1e3 #Umrechnung in Liter

print(' Volumen vom T-Stück welches zum messgerät geht: ', v_tm, '\n')

# Volumen Verbindungsstück zum Messgerät:

r_vm=ufloat( (1/2*12.5)*1e-3, (0.5)*1e-3)
h_vm=ufloat( 60e-3, (0.5)*1e-3)
v_vm=volu( r_vm, h_vm)*1e3 #Umrechnung in Liter

print(' Volumen des Verbindungsstück zum Messgerät', v_vm, '\n')

# Zwischen Volumen mit zwei Kugelventielen
v_zwischen= v_vm + v_tm + v_dk + v_sk + v_gk + v_kbt + 2*v_kv + v_v2t + v_vt + v_t

print(v_zwischen)
## Volumen Drehschieber ( zwei Kugelventile)

print('-----------------------------------------------------------------')
print(' spezifisches Volumen Drehschieber \n-----------------------------------------------------------------\n')

# Volumen großer Schlauch

r_gs=ufloat( (1/2* 23.2)*1e-3, (1/2)*1e-3)
h_gs=ufloat( 118e-2, (1/2)*1e-3)
v_gs=volu(r_gs, h_gs)*1e3 #Umrechnung in Liter

v_dreh= v_gs+v_zwischen

print('Volumen des großen Schlauch (D): ', v_gs, '\n')
print('Gesamtvolumen Drehschieber Methode: ',v_dreh, '\n')

## Volumen Turbo ( drei Kugelventile)

print('-----------------------------------------------------------------')
print('spezifisches Volumen Turbo \n-----------------------------------------------------------------\n')

#Volumen Schlauch zwischen M-Pumpe und Kreuzung
r_st=ufloat( (1/2*17)*1e-3, (1/2)*1e-3)
h_st=ufloat( (21*1e-2), (1/2*1e-3))
v_st=volu(r_st,h_st)*1e3 #Umrechnung in Liter

v_turbo=v_st+v_kv+v_zwischen

print('Volumen Schlauch zur M-Pumpe', v_st ,'\n')

print('Volumen Turbopumpe: ', v_turbo, '\n')
###  Drehschieber
#Konstanten
enddruck=ufloat(1e-2, 1e-2*0.2)


p_druck, t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5 = np.genfromtxt('./messdaten/drehshieber/drehschieber_druck.txt', unpack=True)

preasure_druck=unp.uarray(p_druck,p_druck*0.2)

zeiten_dreh_druck_gemittelt=mittelwert_zeit( t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5)
lograritmierter_druck=unp.log( (preasure_druck-enddruck )/ (preasure_druck[0]-enddruck) )


## Druckkurve
print('-----------------------------------------------------------------')
print(' Auswertung Drehschieber Druckkurve \n-----------------------------------------------------------------\n')


# Geradenfir Bereich
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

test=unp.uarray( noms(lograritmierter_druck), stds(lograritmierter_druck)) # Umschreibung damit es kein Problem mit dem Datentyp gibt: AffineScalarFunc -> Variable
# Tabelle für die Druckkurve mit den Fitgeraden
l.Latexdocument('./table/dreh/druck_messdaten.tex').tabular(
data = [preasure_druck, test, t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5, zeiten_dreh_druck_gemittelt], #Data incl. unpuarray
header = [r'p(t) / \milli\bar', r'\ln( \frac{p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}}}', 't_1 / \second', 't_2 / \second',  't_3 / \second',  't_4 / \second',  't_5 / \second', '\overline{t} / \second'],
places = [(1.1,1.1), (1.1, 1.1), 1, 1, 1, 1, 1, (1.1, 1.1)],
caption = 'Für die Bestimmung des Saugvermögens $S$ der Drehschieberpumpe gemessene Drücke. Die Messung wurde bei Raumtemperatur durchgeführt. Es ist $p_{\mathrm{g}}=\SI{1e-2\pm 2e-4}{\milli\bar}$ der Enddruck und  $p_{\mathrm{g}}=\SI{1e3}{\milli\bar}$',
label = 'druck_dreh')

print(type(lograritmierter_druck[0]), type(test[0]))


## Leckkurve

#Leckkurve für den druck 1mbar

p_dreh_leck_1, t_1_dreh_leck_1, t_2_dreh_leck_2, t_1_dreh_leck_3 = np.genfromtxt('./messdaten/drehshieber/drehschieber_leck_1.txt',unpack=True)


def auswertung_leck(p, t_1, t_2, t_3,name):
    t_gemittelt=mittelwert_zeit_leck(t_1,t_2,t_3)
    preasure=unp.uarray(p,p*0.2)
    messwerte={}
    #print('\n \n', preasure, '\n\n')
    parms, cov = curve_fit(g,noms(t_gemittelt),noms(preasure) )
    error= np.sqrt(np.diag(cov))
    m_u=ufloat(parms[0],error[0])
    b_u=ufloat(parms[1],error[1])
    print(' Steigung der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',m_u )
    print(' y-Achsen der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',b_u )
    print('\n \n \n')
    messwerte['Steigung']=m_u
    messwerte['Achsenabschnitt']=b_u

    laufvariabele=np.linspace(noms(t_gemittelt[0])-1, noms(t_gemittelt[-1])+1,10000)
    plt.clf()
    plt.grid()
    plt.errorbar(noms(t_gemittelt),noms(p), xerr=stds(t_gemittelt), yerr=stds(preasure),fmt='.',label='Messwerte')
    plt.plot(laufvariabele, noms(m_u)* laufvariabele+ noms(b_u), label='Regressionsgerade')
    plt.xlabel(r'$ t \, / \, s $')
    plt.ylabel(r'$ p \, / \, mbar $')
    plt.legend(loc='upper left')
    plt.savefig('./plots/'+ name + '/leckrate_' + name +'_'+ str(noms(p[0])) + '.pdf')

    l.Latexdocument('./table/'+name+'/'+name+'_tabelle_' +str(noms(p[0]))+ '.tex').tabular(
    data = [preasure, t_1,t_2,t_3, t_gemittelt], #Data incl. unpuarray
    header = [r'p / \milli\bar', 't_1 / \second', 't_2 / \second','t_3 / \second', '\overline{t} / \second'],
    places = [(1.1,1.1), 1, 1, 1, (1.1, 1.1)],
    caption = 'Gemessene Drücke bei der Leckkratenmethode für die Drehschieberpumpe mit $p_{\mathrm{l}}=' +str(p[0]) +'$. Messung bei Raumtemperatur.',
    label = 'leck_' + name + '_leck_'+ str(noms(p[0])) + '.pdf' )

    return messwerte

auswertung_leck(p_dreh_leck_1, t_1_dreh_leck_1, t_2_dreh_leck_2, t_1_dreh_leck_3,'dreh')

print('-----------------------------------------------------------------')
print('Auswertug Drehschieber Leckkratenmessung\n-----------------------------------------------------------------\n')

# Auswertung der Leckkurve für p_0=0.8mbar

p_dreh_leck_2, t_1_dreh_leck_2, t_2_dreh_leck_2, t_1_dreh_leck_2 = np.genfromtxt('./messdaten/drehshieber/drehschieber_leck_0.8.txt',unpack=True)

auswertung_leck(p_dreh_leck_2, t_1_dreh_leck_2, t_2_dreh_leck_2, t_1_dreh_leck_2,'dreh')

# Auswertung der Leckkurve für p_0=0.4mbar

p_dreh_leck_3, t_1_dreh_leck_3, t_2_dreh_leck_3, t_1_dreh_leck_3 = np.genfromtxt('./messdaten/drehshieber/drehschieber_leck_0.4.txt',unpack=True)

auswertung_leck(p_dreh_leck_3, t_1_dreh_leck_3, t_2_dreh_leck_3, t_1_dreh_leck_3,'dreh')

# Auswertung der Leckkurve für p_0=0.1mbar /drehschieber

p_dreh_leck_4, t_1_dreh_leck_4, t_2_dreh_leck_4, t_1_dreh_leck_4 = np.genfromtxt('./messdaten/drehshieber/drehschieber_leck_0.1.txt',unpack=True)

auswertung_leck(p_dreh_leck_4, t_1_dreh_leck_4, t_2_dreh_leck_4, t_1_dreh_leck_4,'dreh')




print('-----------------------------------------------------------------')
print('Ab hier Auswertung für die Tubopumpe \n-----------------------------------------------------------------\n')

###  Turbo

## Druckkurve

enddruck_turbo=ufloat(2e-6, 2e-6*0.1)


p_druck_turbo, t_tdruck_1, t_tdruck_2, t_tdruck3, t_tdruck4, t_tdruck5 = np.genfromtxt('./messdaten/turbo/turbo_druck.txt', unpack=True)

preasure_druck_turbo=unp.uarray(p_druck_turbo,p_druck_turbo*0.1)

zeiten_druck_gemittelt_turbo=mittelwert_zeit( t_tdruck_1, t_tdruck_2, t_tdruck3, t_tdruck4, t_tdruck5)

lograritmierter_druck_turbo=unp.log( (preasure_druck_turbo-enddruck_turbo )/ (preasure_druck_turbo[0]-enddruck_turbo))


## zeibereiche 1: [0:7, 2:[6:11], 3:[10:-1]

parms_druck_turbo_1, cov_druck_turbo_1= curve_fit(g,noms(zeiten_druck_gemittelt_turbo[0:7]), noms(lograritmierter_druck_turbo[0:7]) )
error_druck_turbo_1= np.sqrt(np.diag(cov_druck_turbo_1))
m_u_druck_turbo_1=ufloat(parms_druck_turbo_1[0],error_druck_turbo_1[0])
b_u_druck_turbo_1=ufloat(parms_druck_turbo_1[1],error_druck_turbo_1[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 1', m_u_druck_turbo_1)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 1', b_u_druck_turbo_1)
print('\n')
# Geradenfit Bereich 2
parms_druck_turbo_2, cov_druck_turbo_2 = curve_fit(g,noms(zeiten_druck_gemittelt_turbo[6:11]), noms(lograritmierter_druck_turbo[6:11]) )
error_druck_turbo_2= np.sqrt(np.diag(cov_druck_turbo_2))
m_u_druck_turbo_2=ufloat(parms_druck_turbo_2[0],error_druck_turbo_2[0])
b_u_druck_turbo_2=ufloat(parms_druck_turbo_2[1],error_druck_schieber_2[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 2', m_u_druck_turbo_2)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 2', b_u_druck_turbo_2)
print(' \n')
# Geradenfit Bereich 3
parms_druck_turbo_3, cov_druck_turbo_3 = curve_fit(g,noms(zeiten_druck_gemittelt_turbo[-3:14]), noms(lograritmierter_druck_turbo[-3:14]) )
error_druck_turbo_3= np.sqrt(np.diag(cov_druck_turbo_3))
m_u_druck_turbo_3=ufloat(parms_druck_turbo_3[0],error_druck_turbo_3[0])
b_u_druck_turbo_3=ufloat(parms_druck_turbo_3[1],error_druck_turbo_3[1])
print(' Steigung der Druckkurve für die Drehsch im Bereich 3', m_u_druck_turbo_3)
print(' y-Achsen der Druckkurve für die Drehsch im Bereich 3', b_u_druck_turbo_3)
print('\n \n \n')

t_1=np.linspace(noms(zeiten_druck_gemittelt_turbo[0])-1,noms(zeiten_druck_gemittelt_turbo[6]+1/2),1000)
t_2=np.linspace(noms(zeiten_druck_gemittelt_turbo[6])-1,noms(zeiten_druck_gemittelt_turbo[10]+1),1000)
t_3=np.linspace(noms(zeiten_druck_gemittelt_turbo[-3])-1,noms(zeiten_druck_gemittelt_turbo[-1]+1),1000)

plt.clf()
plt.grid()
plt.errorbar(noms(zeiten_druck_gemittelt_turbo),noms(lograritmierter_druck_turbo), xerr=stds(zeiten_druck_gemittelt_turbo), yerr=stds(lograritmierter_druck_turbo),fmt='.',label='Messwerte')
plt.plot(t_1, noms(m_u_druck_turbo_1)* t_1+ noms(b_u_druck_turbo_1), label='Regressionsgerade 1')
plt.plot(t_2, noms(m_u_druck_turbo_2)* t_2+ noms(b_u_druck_turbo_2), label='Regressionsgerade 2')
plt.plot(t_3, noms(m_u_druck_turbo_3)* t_3+ noms(b_u_druck_turbo_3), label='Regressionsgerade 3')
plt.xlabel(r'$ t \, / \, s $')
plt.ylabel(r'$ \ln\left( \frac{ p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}} } \right) $')
plt.legend()
#plt.show()
plt.savefig('./plots/turbo/druckplot_turbo.pdf')

plt.clf()
plt.grid()
plt.errorbar(noms(zeiten_druck_gemittelt_turbo[:-3]),noms(lograritmierter_druck_turbo[:-3]), xerr=stds(zeiten_druck_gemittelt_turbo[:-3]), yerr=stds(lograritmierter_druck_turbo[:-3]),fmt='.',label='Messwerte')
plt.plot(t_1, noms(m_u_druck_turbo_1)* t_1+ noms(b_u_druck_turbo_1), label='Regressionsgerade 1')
plt.plot(t_2, noms(m_u_druck_turbo_2)* t_2+ noms(b_u_druck_turbo_2), label='Regressionsgerade 2')
#plt.plot(t_3, noms(m_u_druck_turbo_3)* t_3+ noms(b_u_druck_turbo_3), label='Regressionsgerade 3')
plt.xlabel(r'$ t \, / \, s $')
plt.ylabel(r'$ \ln\left( \frac{ p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}} } \right) $')
plt.legend()
#plt.show()
plt.savefig('./plots/turbo/druckplot_turbo_zoom.pdf')

test=unp.uarray( noms(lograritmierter_druck_turbo), stds(lograritmierter_druck_turbo)) # Umschreibung damit es kein Problem mit dem Datentyp gibt: AffineScalarFunc -> Variable
l.Latexdocument('./table/turbo/druck_messdaten.tex').tabular(
data = [preasure_druck_turbo, test, t_tdruck_1, t_tdruck_2, t_tdruck3, t_tdruck4, t_tdruck5, zeiten_druck_gemittelt_turbo], #Data incl. unpuarray
header = [r'p(t) / \bar', r'\ln( \frac{p(t)-p_{\mathrm{g}} }{p_0-p_{\mathrm{g}}}', 't_1 / \second', 't_2 / \second',  't_3 / \second',  't_4 / \second',  't_5 / \second', '\overline{t} / \second'],
places = [(1.1,1.1), (1.1, 1.1), 1, 1, 1, 1, 1, (1.1, 1.1)],
caption = 'Für die Bestimmung des Saugvermögens $S$ der Turbopumpe gemessene Drücke. Die Messung wurde bei Raumtemperatur durchgeführt. Es ist $p_{\mathrm{g}}=\SI{2e-6\pm 2e-7}{\milli\bar}$ der Enddruck und  $p_{\mathrm{g}}=\SI{1e-2}{\milli\bar}$',
label = 'druck_turbo')






## Leckkurve
def auswertung_leck_turbo(p, t_1, t_2, t_3,name):
    p*=1e3
    preasure=unp.uarray(p,p*0.4)
    t_gemittelt=mittelwert_zeit_leck(t_1,t_2,t_3)
    messwerte={}
    #print('\n \n', t_gemittelt, '\n\n')
    parms, cov = curve_fit(g,noms(t_gemittelt),p)
    error= np.sqrt(np.diag(cov))
    m_u=ufloat(parms[0],error[0])
    b_u=ufloat(parms[1],error[1])
    print(' Steigung der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',m_u )
    print(' y-Achsen der Druckkurve für die Drehsch im Bereich ', p[0], 'ist: ',b_u )
    print('\n \n \n')
    messwerte['Steigung']=m_u
    messwerte['Achsenabschnitt']=b_u

    laufvariabele=np.linspace(noms(t_gemittelt[0])-1, noms(t_gemittelt[-1])+1,10000)
    plt.clf()
    plt.grid()
    plt.errorbar(noms(t_gemittelt),p, xerr=stds(t_gemittelt), yerr=stds(preasure),fmt='.',label='Messwerte')
    plt.plot(laufvariabele, noms(m_u)* laufvariabele+ noms(b_u), label='Regressionsgerade')
    plt.xlabel(r'$ t \, / \, s $')
    plt.ylabel(r'$ p \, / \, bar $')
    plt.legend()
    plt.savefig('./plots/'+ name + '/leckrate_' + name +'_'+ str(p[0]) + '.pdf')

    l.Latexdocument('./table/'+name+'/'+name+'_tabelle_' +str(p[0])+ '.tex').tabular(
    data = [preasure, t_1,t_2,t_3, t_gemittelt], #Data incl. unpuarray
    header = ['p / \bar', 't_1 / \second', 't_2 / \second','t_3 / \second', '\overline{t} / \second'],
    places = [(1.1,1.1), 1, 1, 1, (1.1, 1.1)],
    caption = 'Gemessene Drücke bei der Leckkratenmethode für die Drehschieberpumpe mit $p_{\mathrm{l}}=' +str(p[0]) +'$. Messung bei Raumtemperatur.',
    label = 'leck_' + name + '_leck_'+ str(p[0]) + '.pdf' )

    return messwerte



# Auswertung der Leckkurve für p_0=1e-4mbar

p_turbo_leck_1, t_1_turbo_leck_1, t_2_turbo_leck_1, t_3_turbo_leck_1 = np.genfromtxt('./messdaten/turbo/turbo_leck_1e-04.txt',unpack=True)

auswertung_leck_turbo(p_turbo_leck_1,t_1_turbo_leck_1, t_2_turbo_leck_1, t_3_turbo_leck_1,'turbo')

# Auswertung der Leckkurve für p_0=2e-4mbar

p_turbo_leck_2, t_1_turbo_leck_2, t_2_turbo_leck_2, t_3_turbo_leck_2 = np.genfromtxt('./messdaten/turbo/turbo_leck_2e-4.txt',unpack=True)

auswertung_leck_turbo(p_turbo_leck_2,t_1_turbo_leck_2, t_2_turbo_leck_2, t_3_turbo_leck_2,'turbo')

# Auswertung der Leckkurve für p_0=3e-4mbar

p_turbo_leck_3, t_1_turbo_leck_3, t_2_turbo_leck_3, t_3_turbo_leck_3 = np.genfromtxt('./messdaten/turbo/turbo_leck_3e-4.txt',unpack=True)

auswertung_leck_turbo(p_turbo_leck_3,t_1_turbo_leck_3, t_2_turbo_leck_3, t_3_turbo_leck_3,'turbo')


# Auswertung der Leckkurve für p_0=5e-5mbar

p_turbo_leck_4, t_1_turbo_leck_4, t_2_turbo_leck_4, t_3_turbo_leck_4 = np.genfromtxt('./messdaten/turbo/turbo_leck_5e-05.txt',unpack=True)

auswertung_leck_turbo(p_turbo_leck_4,t_1_turbo_leck_4, t_2_turbo_leck_4, t_3_turbo_leck_4,'turbo')
