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
import latex as l
r = l.Latexdocument('results.tex')

from scipy.constants import *





##r = l.Latexdocument('results.tex')
#u = UnitRegistry()
#Q_ = u.Quantity
#import pandas as pd
#from pandas import Series, DataFrame
##series = pd.Series(data, index=index)
## d = pd.DataFrame({'colomn': series})
#
#a = ufloat(1, 0.1)
#r.app('I', Q_(a, 'ampere'))
#r.makeresults()
##r.makeresults()
def g(m,x,b):
    return m*x+b

def linfit (x,y,x_min,x_plus,y_min,y_plus,name):
    messwerte={}
    def g(m,x,b):
        return m*x+b

    parms, cov = curve_fit(g,x, y )
    error= np.sqrt(np.diag(cov))
    m_u=ufloat(parms[0],error[0])
    b_u=ufloat(parms[1],error[1])
    messwerte['Steigung']=m_u
    messwerte['Achsenabschnitt']=b_u

    plt.clf()
    plt.xlim(x[0]-x_min,x[-1]+x_plus)
    plt.ylim(y[0]-y_min,y[-1]+y_plus)
    variabel=np.linspace(x[0]-0.5*x[1],x[-1]+0.5*x[1],1000)
    plt.plot(x,y,'rx',label= r'$\mathrm{Spannungspunkte}$')
    plt.plot(variabel,g(parms[0],variabel,parms[1]),'b-',label=r'$\mathrm{Regressionsgerade}$ ')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel(r'$\mathrm{Abstand \, in \, \mathrm{cm}} $',size=25)
    plt.ylabel(r'$\mathrm{Spannung} \,\,  U_{\mathrm{B}} \, \,\mathrm{in \, V}$',size=25)
    #plt.show()
    plt.savefig( name +'.pdf')
    return messwerte






def weglange (temp):
    temp_k=temp+237.15  #kelvin
    p=5.5*1e7*np.exp(-6876/temp_k) #mbar
    w=0.0029/p #cm
    verhaeltnis=1/w
    return {'Temperatur_k':temp_k,'Druck':p,'Weglange':w,'verhaeltnis':verhaeltnis}






####### Frank-Hertz-Kurve

abstand_frank_hertz, spannung_frank_hertz=np.genfromtxt('spannung_abstand_frankhertzkurve_188grad.txt',unpack=True)
anzahl,abstaende_maxima=np.genfromtxt('abstaende_frankhertzkurve_188.txt',unpack=True)


parmeter_frank_hertz=linfit(abstand_frank_hertz,spannung_frank_hertz,1,1,1,1,'frank_hertz_kuvre')
anregungsspannung=g(noms(parmeter_frank_hertz['Steigung']),abstaende_maxima,noms(parmeter_frank_hertz['Achsenabschnitt']))
anregungspannung_u=ufloat(np.mean(anregungsspannung), np.std(anregungsspannung, ddof=1) / np.sqrt(len(anregungsspannung)))
anregungsenergie=anregungspannung_u  #eV

print('\n \n \n')
print('Parameter Spannungs Fit Frank (m,b)',parmeter_frank_hertz['Steigung'],parmeter_frank_hertz['Achsenabschnitt'])
print('Anregungsenergie (eV)',anregungsenergie)
wellenlaenge=(h*speed_of_light)/(anregungsenergie*e)

print('Wellenlänge',wellenlaenge)
K_frank=2*anregungspannung_u-g(noms(parmeter_frank_hertz['Steigung']),3.1,noms(parmeter_frank_hertz['Achsenabschnitt']))
print('K frankherz',K_frank)
weglange_frank=weglange(188.0)
print('Verhaltnis w/a Frank-Hertz, 188',weglange_frank['verhaeltnis'])
print('\n \n \n')



##Tabelle Abstand Spannung für Frank Hertz
l.Latexdocument('spannungen_abstand_frank.tex').tabular([abstand_frank_hertz,spannung_frank_hertz],
'{Abstand in $\si{\\centi\\meter}$} & {Spannung in $\si{\\volt}$}', [1, 0] ,
caption = 'Aus Abbildung \\ref{} abgelesene Spannung-Abstandspare.', label = 'spannung_abstand_frank')

##Tabelle Abstand zwischen zwei Maxima

l.Latexdocument('abstaende_frank.tex').tabular([anzahl,abstaende_maxima,anregungsspannung],
'{Nummerierung} & {Abstand in $\si{\\centi\\meter}$} & {Abstand in $\si{\\eV}$}', [0, 1,2] ,
caption = 'Aus Abbildung \\ref{} abgelesene Abstände der Maxima.', label = 'abstand_maxima')



#### Steigungsdreieckauswertung für die Energieverteilung bei T=28 Grad
def steigung(x_1,x_2,dy):
    dx=x_2-x_1
    messpunkt=x_1+dx/2
    steigung=dy/dx
    #print(pd.DataFrame({'X1':x_1,'X2':x_2, 'dY':dy, 'Steigung':steigung}))
    return {'Messpunkt': messpunkt, 'Steigung': steigung}



x_1_zim,x_2_zim,dy_zim=np.genfromtxt('steigung_energieverteilung_28grad.txt',unpack=True)

abstand_zim, spanung_zim=np.genfromtxt('spannungen_abstand_energievert_28_grad.txt',unpack=True)
parmeter_zim=linfit(abstand_zim,spanung_zim,1,1,1,1,'zim')

steigungen_zim=steigung(x_1_zim,x_2_zim,dy_zim)
spannung_zim=g(noms(parmeter_zim['Steigung']),steigungen_zim['Messpunkt'],noms(parmeter_zim['Achsenabschnitt']))

K_zimmer=11-spannung_zim[np.argmax(steigungen_zim['Steigung'])]
weglange_zim=weglange(28.0)
print('Bestimmtes K_zimmer',K_zimmer)
print('Verhältnis w/a Zimmertemp, 28', weglange_zim['verhaeltnis'])
print('\n\n\n')

plt.clf()
plt.ylim(0,170)
plt.ylabel(r'$\mathrm{Steigung}$')
plt.xlabel(r'$U_{\mathrm{a}}\,\mathrm{in \, V}$')
plt.plot(spannung_zim,steigungen_zim['Steigung'],'rx',label=r'$\mathrm{\mathrm{Wert\, des\, Steigungsdreieckes}}$')
plt.legend(loc='best')
plt.savefig('energie_zim.pdf')

##Tabelle Spannungen-Abstände Energerverteilung bei  Zimmertemp
l.Latexdocument('spannungen_abstand_energie_zimmer.tex').tabular([abstand_zim,spanung_zim],
'{Abstand in $\si{\\centi\\meter}$} & {Spannung in $\si{\\volt}$}', [1, 0] ,
caption = 'Aus Abbildung \\ref{} abgelesene Spannung-Abstandspaare.', label = 'spannung_abstand_zim')

##Tabelle ResultatSteigungsdreiecke für die Energieverteilung bei Zimmertemp
l.Latexdocument('steigungen_energie_zimmer.tex').tabular([x_1_zim,x_2_zim,dy_zim,steigungen_zim['Steigung'],steigungen_zim['Messpunkt'],spannung_zim],
'{$x_1$ in $\si{\\centi\\meter}$} & {$x_2$ in $\si{\\centi\\meter}$} & { ${\Delta y}$ in $\si{\\milli\\meter}$} & {$\\frac{\Delta y}{\Delta x}$ in \si{\\centi\\meter\per\\milli\\meter}} & {Messpunkt in $\si{\\centi\\meter}$} & {Messpunkt in $\si{\\volt}$}', [1, 1,1,2,2,2] ,
caption = 'Aus Abbildung \\ref{} abgelesene Steigungen.', label = 'steigungen_zim')



#### Ioniersungsspannung
abstand_ioni, spannung_ioni=np.genfromtxt('abstand_spannung_ionisierung_104grad.txt',unpack=True)

parmeter_ioni=linfit(abstand_ioni,spannung_ioni,1,1,1,1,'ioni')

###Messwerte_für_die_Sekante/Tangente
X=[13.9,16.6,18.7,19.6] ##Gewählte Messpunkte
Y=[3.5,7.0,9.7,10.9]


k_u=ufloat(np.mean(np.array([float(noms(K_frank)),K_zimmer])), np.std(np.array([float(noms(K_frank)),K_zimmer]), ddof=1) / np.sqrt(len(np.array([float(noms(K_frank)),K_zimmer]))))
params_gerade_io=linfit(X,Y,1,1,1,1,'gerade_io')
nulldurchgang=-params_gerade_io['Achsenabschnitt']/params_gerade_io['Steigung']-k_u
weglange_ioni=weglange(104.0)


print('K Mittelwert ',k_u)
print('Verhaltnis w/a Frank-Hertz, 104',weglange_ioni['verhaeltnis'])
print('Ionisierungsenergie ',nulldurchgang)
print('\n\n\n')


##Tabelle Abstand-Spannung für Ionisationspannung
l.Latexdocument('spannungen_abstand_ioni.tex').tabular([abstand_ioni,spannung_ioni],
'{Abstand in $\si{\\centi\\meter}$} & {Spannung in $\si{\\volt}$}', [1, 0] ,
caption = 'Aus Abbildung \\ref{} abgelesene Spannung-Abstandspaare.', label = 'spannung_abstand_ioni')




#### Steigungsdreieckauswertung für die Energieverteilung bei T=150 Grad

abstand_hot,spanung_hot=np.genfromtxt('abstand_spannung_energieverteilung_150grad.txt',unpack=True)
x_1_hot,x_2_hot,dy_hot=np.genfromtxt('steigung_energieverteilung_150grad.txt',unpack=True)

steigungen_hot=steigung(x_1_hot,x_2_hot,dy_hot)
weglange_hot=weglange(150.0)
parmeter_hot=linfit(abstand_hot,spanung_hot,1,1,1,1,'spannungsfit_energieverteilung_150grad')
spannung_hot=g(noms(parmeter_hot['Steigung']),steigungen_hot['Messpunkt'],noms(parmeter_hot['Achsenabschnitt']))
print('Verhätnis w/a Hot, 150',weglange_hot['verhaeltnis'])

plt.clf()
plt.ylabel(r'$\mathrm{Steigung}$')
plt.xlabel(r'$U_{\mathrm{a}}\,\mathrm{in \, V}$')
plt.plot(spannung_hot,steigungen_hot['Steigung'],'rx',label=r'$\mathrm{Wert\, des\, Steigungsdreieckes}$')
plt.legend(loc='best')
plt.savefig('energie_hot.pdf')

##Tabelle Spannungen-Abstände Energerverteilung bei  150 Grad
l.Latexdocument('spannungen_abstand_energie_hot.tex').tabular([abstand_hot,spanung_hot],
'{Abstand in $\si{\\centi\\meter}$} & {Spannung in $\si{\\volt}$}', [1, 0] ,
caption = 'Aus Abbildung \\ref{} abgelesene Spannung-Abstandspaare.', label = 'spannung_abstand_hot')

##Tabelle ResultatSteigungsdreiecke für die Energieverteilung bei 150 Grad
l.Latexdocument('steigungen_energie_150grad.tex').tabular([x_1_hot,x_2_hot,dy_hot,steigungen_hot['Steigung'],steigungen_hot['Messpunkt'],spannung_hot],
'{$x_1$ in $\si{\\centi\\meter}$} & {$x_2$ in $\si{\\centi\\meter}$} & { ${\Delta y}$ in $\si{\\milli\\meter}$} & {$\\frac{\Delta y}{\Delta x}$ in \si{\\centi\\meter\per\\milli\\meter}} & {Messpunkt in $\si{\\centi\\meter}$} & {Messpunkt in $\si{\\volt}$}', [1, 1,1,2,2,2] ,
caption = 'Aus Abbildung \\ref{} abgelesene Steigungen.', label = 'steigungen_hot')





#########Tabellen

##Sapnnugnsfit
liste_steigungen=[ float(noms(parmeter_zim ['Steigung'])), float(noms(parmeter_hot ['Steigung'])), float(noms(parmeter_frank_hertz['Steigung'])),float(noms(parmeter_ioni['Steigung']))]
liste_steigungen_fehler=[float(stds(parmeter_zim ['Steigung'])),float(stds(parmeter_hot ['Steigung'])),float(stds(parmeter_frank_hertz['Steigung'])),float(stds(parmeter_ioni['Steigung']))]
liste_abschnitt=[float(noms(parmeter_zim ['Achsenabschnitt'])),float(noms(parmeter_zim ['Achsenabschnitt'])),float(noms(parmeter_frank_hertz['Achsenabschnitt'])),float(noms(parmeter_ioni['Achsenabschnitt']))]
liste_abschnitt_fehler=[float(stds(parmeter_zim ['Achsenabschnitt'])),float(stds(parmeter_hot ['Achsenabschnitt'])),float(stds(parmeter_frank_hertz['Achsenabschnitt'])),float(stds(parmeter_ioni['Achsenabschnitt']))]
liste=[1,2,3,4]
l.Latexdocument('spannungsparameter.tex').tabular([liste,liste_steigungen,liste_steigungen_fehler,liste_abschnitt,liste_abschnitt_fehler],
'{Versuchsteil} & { $m$ in $\si{\\volt\\centi\\meter\\per}$} & {$\sigma_\mathrm{m}$ in $\si{\\volt\\centi\\per\\meter}$} & {$b$ in $\si{\\volt}$} & {$\sigma_\mathrm{b}$ in $\si{\\volt}$}', [0,3, 3, 2,2] ,
caption = 'Regressiongerade für die Abstand in Spannungs Umrechnung. Im Versuchsteil $1$ wird die Energieverteilung bei $T=\SI{28}{\\celsius}$ untersucht, $2$ umfassst die Untersuchung der Energieverteilung bei $T=\SI{155}{\\celsius}$, der dritte Abschnitt $(3)$ beschäftigt sich mit der Analyse der Frank-Hertz-Kurve und im Abschnitt $4$ wird die Ionisierungsspannung bestimmt.', label = 'umrech')

### Freieweglängen
T=[28.0,104.0,150.0,188.0]
T_k=T+273.15*np.ones(len(T))
p=5.5*1e7*np.exp(-6876/T_k)



l.Latexdocument('weglange.tex').tabular([T,T+273.15*np.ones(len(T)),p,0.0029/p,p/0.0029],
'{$T$ in $\si{\\celsius}$} & {$T$ in $\si{\\kelvin}$} & {$p_{\mathrm{sät}} in $\si{\\milli\\bar}$} & {$\overline{w}$ in $\si{\\centi\\meter}$} & {$\\frac{a/w}$}', [0, 2, 3,4, 0] ,
caption = 'Ergebnisse für die Verhältnisberechenung $a\/w$.', label = 'weg')
