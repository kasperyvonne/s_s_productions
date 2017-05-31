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





#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

a = ufloat(1, 0.1)
r.app('I', Q_(a, 'ampere'))
r.makeresults()
#r.makeresults()
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
print('Verhaltnis w/a Frank-Hertz',weglange_frank['verhaeltnis'])
print('\n \n \n')





#### Steigungsdreieckauswertung für die Energieverteilung bei T=28 Grad
def steigung(x_1,x_2,dy):
    dx=x_2-x_1
    messpunkt=x_1+dx/2
    steigung=dy/dx
    #print(pd.DataFrame({'X1':x_1,'X2':x_2, 'dY':dy, 'Steigung':steigung}))
    return {'Messpunkt': messpunkt, 'Steigung': steigung}

x_1_zim,x_2_zim,dy_zim=np.genfromtxt('steigung_energieverteilung_28grad.txt',unpack=True)

abstand_zim, spannung_zim=np.genfromtxt('spannungen_abstand_energievert_28_grad.txt',unpack=True)
parmeter_zim=linfit(abstand_zim,spannung_zim,1,1,1,1,'zim')

steigungen_zim=steigung(x_1_zim,x_2_zim,dy_zim)
spannung_zim=g(noms(parmeter_zim['Steigung']),steigungen_zim['Messpunkt'],noms(parmeter_zim['Achsenabschnitt']))

K_zimmer=11-spannung_zim[np.argmax(steigungen_zim['Steigung'])]
print('Bestimmtes K_zimmer',K_zimmer)

plt.clf()
plt.ylim(0,170)
plt.ylabel(r'$\mathrm{Steigung}$')
plt.xlabel(r'$U_{\mathrm{a}}\,\mathrm{in \, V}$')
plt.plot(spannung_zim,steigungen_zim['Steigung'],'rx',label=r'$\mathrm{\mathrm{Wert\, des\, Steigungsdreieckes}}$')
plt.legend(loc='best')
plt.savefig('energie_zim.pdf')







#### Ioniersungsspannung
abstand_ioni, spannung_ioni=np.genfromtxt('abstand_spannung_ionisierung_104grad.txt',unpack=True)

parmeter_ioni=linfit(abstand_ioni,spannung_ioni,1,1,1,1,'ioni')

###Messwerte_für_die_Sekante/Tangente
X=[13.9,16.6,18.7,19.6] ##Gewählte Messpunkte
Y=[3.5,7.0,9.7,10.9]


params_gerade_io=linfit(X,Y,1,1,1,1,'gerade_io')
nulldurchgang=-params_gerade_io['Achsenabschnitt']/params_gerade_io['Steigung']-K_frank
weglange_ioni=weglange(104.0)

print('Verhaltnis w/a Frank-Hertz',weglange_ioni['verhaeltnis'])
print('Ionisierungsenergie !!! Noch mit K Frank!!!',nulldurchgang)
print('\n\n\n')




#### Steigungsdreieckauswertung für die Energieverteilung bei T=150 Grad

x_1_hot,x_2_hot,dy_hot=np.genfromtxt('steigung_energieverteilung_150grad.txt',unpack=True)

steigungen_hot=steigung(x_1_hot,x_2_hot,dy_hot)

plt.clf()
plt.ylabel(r'$\mathrm{Steigung}$')
plt.xlabel(r'$U_{\mathrm{a}}\,\mathrm{in \, V}$')
plt.plot(steigungen_hot['Messpunkt'],steigungen_hot['Steigung'],'rx',label=r'$\mathrm{\mathrm{Wert\, des\, Steigungsdreieckes}}$')
plt.legend(loc='best')
plt.savefig('energie_hot.pdf')
