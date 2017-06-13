import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math as m
from scipy.constants import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})
#r.makeresults()

fine_structure=0.0072973525664
Ryd=13.605693009


###### Versuchvorberitung

def brag_wink (E,n):
    lam= c * h / ((E*e)) # eventuell noch umrechnen
    winkel= np.degrees(np.arcsin( (n*lam) / (2*201.4*1e-12) ))
    return winkel

def brag_well (winkel, n):
    lam= 2*201.4e-12 * np.sin( m.radians(winkel) ) / n #Eventuell rad to deg
    #return h* c/ (lam*e)
    return lam

def brag_ener (winkel, n):
    lam= 2*201.4e-12 * np.sin( m.radians(winkel) ) / n #Eventuell rad to deg
    return h* c/ (lam*e)


def abschirm(Z, delta_E):
    return Z - ( np.sqrt( (4 /fine_structure ) * np.sqrt( delta_E / Ryd) -  ( 5 * delta_E ) / Ryd ) ) * ( np.sqrt( 1 + (19/32) * ( fine_structure )**2 * ( delta_E / Ryd ) ) )

def absorbkoe(E,Z):
    return Z- np.sqrt(E/Ryd)

energien=np.array([9.65,10.37,13.48,15.21,16.12,18.00,19.00,11.925,13.739])*1e3 ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)
Z= np.array([30,32,35,37,38,40,41,79,79])  ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)



print('\n')
print(' Abschirmung', abschirm(Z,energien))
print('\n')
print('Bragg Winkel',brag_wink(energien,1))
print('\n \n \n')





### Überprüfung der Brag Bedingung:

## Messdaten eingaben

winkel, inten=np.genfromtxt('braggbed.txt',unpack=True)


## Funktion
def auswertung_brag(winkel, inten,x_p,x_m,y_p,y_m,name):
    l.Latexdocument(name+'.tex').tabular([winkel,inten],
    '{$\\alpha_{\mathrm{Gl}} \, / \, \si{\\degree}$} & {$I \, / \, \mathmrm{Imp}/\mathrm{s}$}', [1, 1] ,
    caption = 'Messwerte bei der Untersuchung der Bragg Bedingung.', label = 'bragg_test')
    plt.clf()
    plt.plot(winkel,inten,'rx',label=r'$Intensität$')
    plt.axvline(winkel[np.argmax(inten)],ls='--', color='b',label=r'$Maximum$')
    plt.grid()
    plt.xlabel(r'$2 \theta \, \mathrm{in} \, \mathrm{deg}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.xlim(winkel[0]-x_m,winkel[-1]+x_p)
    plt.ylim(min(inten)-y_m,max(inten)+y_p)
    plt.legend(loc='best')
    plt.savefig(name + '.pdf')

    return winkel[np.argmax(inten)]

print('Bragg Winkel',auswertung_brag(winkel,inten,0.5,0.5,10,1,'bragbed'))


### Emessionspektrum von Kupfer


##Eingabe der Messwerte
winkel_emi, inten_emi= np.genfromtxt('emission_cu.txt',unpack=True)

print(len(winkel_emi))
plt.clf()
plt.plot(winkel_emi, inten_emi)


##Auswertung
def peak(int_1,int_2,winkel,imp ,x_p,x_m,y_p,y_m,name):
    dic={}
    ###Peak detect
    max_1= max( imp [int_1[0] : int_1[1] ] )
    winkel_1=winkel[np.where(imp==max_1)]

    max_2=max(imp[int_2[0]:int_2[1]])
    winkel_2=winkel[np.where(imp==max_2)]

    plt.clf()
    plt.plot(winkel,imp,'rx',label=r'$Intensität$')
    plt.axvline(winkel_2,ls='--', color='b',label=r'$K_{\alpha}$')
    plt.axvline(winkel_1,ls='--', color='g',label=r'$K_{\beta}$')
    plt.grid()
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{deg}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.xlim(winkel[0]-x_m,winkel[-1]+x_p)
    plt.ylim(min(imp)-y_m,max(imp)+y_p)
    plt.legend(loc='best')
    plt.savefig(name + '.pdf')

    l.Latexdocument(name+'.tex').tabular([winkel,imp],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathmrm{Imp}/\mathrm{s}$}', [1, 1] ,
    caption = 'Messwerte bei der Untersuchung des Emmissionspektrum von $\ce{Cu}$.', label = 'emi_cu')

    ###grenzwinkel
    lam_1=brag_well(winkel_1,1)
    lam_2=brag_well(winkel_2,1)
    dic['Wellenlänge']=lam_1
    e_1=brag_ener(winkel_1,1)
    e_2=brag_ener(winkel_2,1)
    dic['Energie_k_beta']=e_1
    dic['Energie_k_alpha']=e_2
    sigma_1=absorbkoe(e_1,29)
    sigma_2=absorbkoe(-e_2+e_1 ,29)

    print('Betrachte: ', name)
    print('\n')
    print('Energie k_beta', e_1)
    print('Energie k_beta', e_2)
    print('Sigma 1', sigma_1)
    print('Sigma 2', sigma_2)
    print('\n\n\n')
    ##Es fehlt die Halbwertsbreite und Abschirmkonstante

peak( [74,87], [86,99], 0.5*winkel_emi,inten_emi,1,1,100,100,'emission_cu')








### Absorptionsspektrum

def absorb(a,b,Z,winkel,imp ,x_p,x_m,y_p,y_m,name):
    ###Peak detect
    maxi=max(imp[a:b])
    winkel_max=winkel[a:b][np.where(imp[a:b]==maxi)]

    mini=min(imp[a:b])
    winkel_min=winkel[a:b][np.where(imp[a:b]==mini)]
    kante=0.5*(winkel_max-winkel_min)
    energie_k=brag_ener(kante,1)
    print(winkel_min,winkel_max)
    print(mini,maxi)
    sigma=absorbkoe(energie_k,Z)

    print('Betrachte: ', name)
    print('\n')
    print('Winkel K Kante', kante)
    print('Energie K Kante', energie_k)
    print('Abschirmkoef', sigma)
    print('\n\n\n')

    plt.clf()
    plt.plot(winkel,imp,'rx',label=r'$Intensität$')
    plt.axvline(winkel_max,ls='--', color='b',label=r'$\theta_{\mathrm{max}}$')
    plt.axvline(winkel_min,ls='--', color='g',label=r'$\theta_{\mathrm{min}}$')
    plt.grid()
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{rad}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.xlim(winkel[0]-x_m,winkel[-1]+x_p)
    plt.ylim(min(imp)-y_m,max(imp)+y_p)
    plt.legend(loc='best')
    plt.savefig(name + '.pdf')

    l.Latexdocument(name+'.tex').tabular([winkel,imp],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathmrm{Imp}/\mathrm{s}$}', [1, 1] ,
    caption = 'Messwerte bei der Untersuchung des Emmissionspektrum von $\ce{Cu}$.', label = 'emi_cu')
    return energie_k
    ####grenzwinkel


##Dateneinlesen

winkel_zirkonium, int_zirkonium = np.genfromtxt('zirkonium_ab.txt',unpack=True)
winkel_germanium, int_germanium = np.genfromtxt('germanium_ab.txt',unpack=True)
winkel_zink, int_zink = np.genfromtxt('zink_ab.txt',unpack=True)
winkel_brom, int_brom = np.genfromtxt('brom_ab.txt',unpack=True)
winkel_stom, int_strom = np.genfromtxt('strontium_ab.txt',unpack=True)

##Plots



##Auswertung

absorb(12,18,40,0.5*winkel_zirkonium,int_zirkonium,0.5,0.5,10,1,'zr')
absorb(1,2,32,0.5*winkel_germanium,int_germanium,1,1,1,1,'germanium')
#absorb(1,2,30,0.5*winkel_zink,int_zink,1,1,1,1,'zink')
#absorb(1,2,35,0.5*winkel_brom,int_brom,1,1,1,1,'brom')
#absorb(1,2,38,0.5*winkel_stom,int_strom,1,1,1,1,'strom')

plt.clf()

#plt.plot(0.5*winkel_zirkonium,int_zirkonium,'rx')

plt.plot(0.5*winkel_germanium,int_germanium)

##plt.plot(0.5*winkel_zink,int_zink)

##plt.plot(0.5*winkel_brom,int_brom)

##plt.plot(0.5*winkel_stom,int_strom)

plt.show()
