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
    return h*c / (e*lam)

def abschirm(Z, delta_E):
    return Z - ( np.sqrt( (4 /fine_structure ) * np.sqrt( delta_E / Ryd) -  ( 5 * delta_E ) / Ryd ) ) * ( np.sqrt( 1 + (19/32) * ( fine_structure )**2 * ( delta_E / Ryd ) ) )

def absorbkoe(E,Z):
    return Z- np.sqrt(E/Ryd)

energien=np.array([9.65,10.37,13.48,15.21,16.12,18.00,19.00,11.925,13.739])*1e3 ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)
Z= np.array([30,32,35,37,38,40,41,79,79])  ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)


winkeli=brag_wink(energien,1)
print('\n')
print(' Abschirmung', abschirm(Z,energien))
print('\n')
print('Bragg Winkel',brag_wink(energien,1))
print('\n')
print('Bragg Winkel',brag_ener(winkeli[3],1))
print('\n')
print('abschirm', absorbkoe(energien,Z))

print('\n \n \n')





### Überprüfung der Brag Bedingung:

## Messdaten eingaben

winkel, inten=np.genfromtxt('braggbed.txt',unpack=True)


## Funktion
def auswertung_brag(winkel, inten,x_p,x_m,y_p,y_m,name):
    l.Latexdocument(name+'.tex').tabular([winkel[0:len(winkel)//2],inten[0:len(winkel)//2], winkel[len(winkel)//2+1:],inten[len(winkel)//2+1:]],
    '{$\\alpha_{\mathrm{Gl}} \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$} & {$\\alpha_{\mathrm{Gl}} \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$}', [1, 0, 1,0] ,
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
    plt.xlim(winkel[0]-x_m,winkel[-2]+x_p)
    plt.ylim(min(imp)-y_m,max(imp)+y_p)
    plt.legend(loc='best')
    plt.savefig(name + '.pdf')
    print(len(winkel)//3)
    l.Latexdocument(name+'.tex').tabular([winkel[0:len(winkel)//3],imp[0:len(winkel)//3], winkel[len(winkel)//3:len(winkel)//3*2],imp[len(winkel)//3:len(winkel)//3 *2],winkel[len(winkel)//3*2:],imp[len(winkel)//3 *2:]],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$} & {$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$} &{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$}', [2, 0 ,2, 0, 2, 0] ,
    caption = 'Messwerte bei der Untersuchung des Emmissionspektrums von $\ce{Cu}$.', label = 'emi_cu')

    ###grenzwinkel
    lam_1=brag_well(winkel_1,1)
    lam_2=brag_well(winkel_2,1)
    dic['Wellenlänge']=lam_1
    e_1=brag_ener(winkel_1,1)
    e_2=brag_ener(winkel_2,1)
    dic['Energie_k_beta']=e_1
    dic['Energie_k_alpha']=e_2
    sigma_1=absorbkoe(e_1,29)
    sigma_2=absorbkoe(4*(-e_2+e_1) ,29)


    print('Betrachte: ', name)
    print('Energie k_beta', e_1)
    print('Energie k_alpha', e_2)
    print('Sigma 1', sigma_1)
    print('Sigma 2', sigma_2)
    print('Maximale Energie',brag_ener(winkel[0],1))
    print('\n\n\n')


    ## Bestimmung der Halbwertsbreite

    #K_alpha peak, Winkel 1

    def g(m,x,b):
        return m*x+b

    imp_winkel1_partner=imp[np.where(imp==max_1)[0][0]-1]
    winkel_1_fit_partner=winkel[np.where(imp==max_1)[0][0]-1]

    parms_winkel1, cov_winkel1 = curve_fit( g ,[winkel_1_fit_partner,winkel_1[0]], [imp_winkel1_partner,imp[np.where(imp==max_1)][0]] )
    print(cov_winkel1)
    error_winkel1= np.sqrt(np.diag(cov_winkel1))
    m_uwinkel1 =ufloat(parms_winkel1[0],error_winkel1[0])
    b_uwinkel1 =ufloat(parms_winkel1[1],error_winkel1[1])

    breite_winkel1=winkel_1-(max_1*0.5-b_uwinkel1)/m_uwinkel1

    print('K_beta fit Steigung',m_uwinkel1)
    print('k_beta fit y_achsenabschnitt', b_uwinkel1)
    print('ort des Maximums', winkel_1)
    print('Höhe des Maximums', max_1)
    print('Partner Ort' ,winkel_1_fit_partner)
    print('partner Höhe',imp_winkel1_partner )
    print('Ort der Halben beta', (max_1*0.5-b_uwinkel1)/m_uwinkel1)
    print('k_beta Breite 1', breite_winkel1)
    print('\n\n\n')

    ###k_alpha peak
    imp_winkel2_partner=imp[np.where(imp==max_2)[0][0]-1]
    winkel_2_fit_partner=winkel[np.where(imp==max_2)[0][0]-1]
    parms_winkel2, cov_winkel2 = curve_fit( g ,[winkel_2_fit_partner,winkel_2[0]], [imp_winkel2_partner,imp[np.where(imp==max_2)][0]])
    error_winkel2= np.sqrt(np.diag(cov_winkel2))
    m_uwinkel2 =ufloat(parms_winkel2[0],error_winkel2[0])
    b_uwinkel2 =ufloat(parms_winkel2[1],error_winkel2[1])

    breite_winkel2=winkel_2-(max_2*0.5-b_uwinkel2)/m_uwinkel2

    print('K_alpha fit Steigung',m_uwinkel2)
    print('k_alpha fit y_achsenabschnitt', b_uwinkel2)
    print('ort des Maximums', winkel_2)
    print('Höhe des Maximums', max_2)
    print('Partner Ort' ,winkel_2_fit_partner)
    print('partner Höhe',imp_winkel2_partner )
    print('Ort der Halben alpha', (max_2*0.5-b_uwinkel2)/m_uwinkel2)
    print('k_alpha Breite 2', breite_winkel2)
    print('\n\n\n')


    ## Plots für die Halbwertsbrete
    lauf1=np.linspace(19,22,1000)
    lauf2=np.linspace(21,23,1000)

    plt.clf()
    plt.plot(winkel,imp,'rx',label=r'$Intensität$')
    plt.plot(lauf1,g(parms_winkel1[0],lauf1,parms_winkel1[1]),'c-',label=r'$Ausgleichgerade K_{\beta}$')
    plt.plot(lauf2,g(parms_winkel2[0],lauf2,parms_winkel2[1]),'y-',label=r'$Ausgleichgerade K_{\alpha}$')
    plt.axvline(winkel_2,ls='--', color='b',label=r'$K_{\alpha}$')
    plt.axvline(winkel_1,ls='--', color='g',label=r'$K_{\beta}$')
    plt.grid()
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{deg}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.xlim(17.5,23.5)
    plt.ylim(0,4400)
    plt.legend(loc='best')
    plt.savefig(name + '_zoom.pdf')

peak( [74,87], [86,99], 0.5*winkel_emi,inten_emi,1,1,100,100,'emission_cu')




### Absorptionsspektrum

def absorb(a,b,Z,winkel,imp ,x_p,x_m,y_p,y_m,name):
    ###Peak detect
    maxi=max(imp[a:b])
    winkel_max=winkel[a:b][np.where(imp[a:b]==maxi)]

    mini=min(imp[a:b])
    winkel_min=winkel[a:b][np.where(imp[a:b]==mini)]
    kante=winkel_min+0.5*(winkel_max-winkel_min)
    energie_k=brag_ener(kante,1)*1e-3
    sigma=absorbkoe(energie_k*1e3,Z)

    print('Betrachte: ', name)
    print('winekl min',winkel_min)
    print('winkel_max',winkel_max)
    print('Winkel K Kante', kante)
    print('Energie K Kante', energie_k)
    print('Abschirmkoef', sigma)
    print('\n\n\n')

    plt.clf()
    plt.plot(winkel,imp,'rx',label=r'$Intensität$')
    plt.axvline(winkel_max,ls='--', color='b',label=r'$\theta_{\mathrm{max}}$')
    plt.axvline(winkel_min,ls='--', color='g',label=r'$\theta_{\mathrm{min}}$')
    plt.grid()
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{deg}$',size=25)
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$',size=25)
    plt.xlim(winkel[0]-x_m,winkel[-1]+x_p)
    plt.ylim(min(imp)-y_m,max(imp)+y_p)
    plt.legend(loc='best',fontsize=20)
    plt.savefig(name + '.pdf')

    l.Latexdocument(name+'.tex').tabular([winkel,imp],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$}', [2, 0] ,
    caption = 'Messwerte bei der Untersuchung des Emmissionspektrums von $\ce{Cu}$.', label = name)
    return energie_k
    ####grenzwinkel


#####Sonder Funktion Zink für subplot

def absorb_zink(a,b,c,d,Z,winkel,imp ,x_p,x_m,y_p,y_m,name):
    ###Peak detect

    kante=winkel[c+2]+0.5*(winkel[c+6]-winkel[c+2])
    energie_k=brag_ener(kante,1)*1e-3
    sigma=absorbkoe(energie_k*1e3,Z)

    print('Betrachte: ', name)
    print('winkel_min',winkel[c+2])
    print('winkel_max',winkel[c+6])
    print('Winkel K Kante', kante)
    print('Energie K Kante', energie_k)
    print('Abschirmkoef', sigma)
    print('\n\n\n')

    plt.clf()
    plt.plot(winkel,imp,'rx',label=r'$Intensität$')
    #plt.axvline(winkel_max,ls='--', color='b',label=r'$\theta_{\mathrm{max}}$')
    #plt.axvline(winkel_min,ls='--', color='g',label=r'$\theta_{\mathrm{min}}$')
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{deg}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.grid()
    plt.xlim(winkel[0]-x_m,winkel[-1]+x_p)
    plt.ylim(min(imp)-y_m,max(imp)+y_p)
    plt.legend(loc='best',)

    plt.axes([0.207,0.375, 0.3,0.3])
    plt.plot(winkel[c:d],imp[c:d],'rx')
    plt.title(r'$\mathrm{Vergrößerung \, der \, K-Kante}$')
    plt.xlim(winkel[c]-x_m,winkel[d]+x_p)
    plt.axvline(winkel[c+6],ls='--', color='b',label=r'$\theta_{\mathrm{max}}$')
    plt.axvline(winkel[c+2],ls='--', color='g',label=r'$\theta_{\mathrm{min}}$')
    plt.ylim(imp[c]-15,imp[d]-20)
    plt.legend(loc='best',fontsize=12)
    plt.grid()

    #plt.xticks([0, -0.1 , -0.2, -0.3], ['0', '-0.1', '-0.2', '-0.3'])
    #plt.yticks([0, 0.02 , 0.04 , 0.06,0.08, 0.1, 0.12, 0.14 ], ['0', '0.02', '0.04','0.06', '0.08' , '0.1', '0.12', '0.14'])


    plt.savefig(name + '.pdf')

    l.Latexdocument(name+'.tex').tabular([winkel,imp],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$}', [2, 0] ,
    caption = 'Messwerte bei der Untersuchung des Absorptionsspektrums von $\ce{Zn}$.', label = name)
    return energie_k


### Sonder Funktion gold

def absorb_gold(a,b,Z,winkel,imp ,x_p,x_m,y_p,y_m,name):
    ###Peak detect
    winkel_l2=winkel[17]
    winkel_l3=winkel[39]

    energie_l2=brag_ener(winkel_l2,1)*1e-3
    energie_l3=brag_ener(winkel_l3,1)*1e-3
    delta_e=(energie_l2-energie_l3)
    abschirm_kosnt=abschirm(Z,delta_e*1e3)
    print('Betrachte: ', name)
    print('Energie L2 Kante',energie_l2)
    print('Energie L3 kante', energie_l3)
    print('Abschirm Konstante', abschirm_kosnt)
    print('\n\n\n')

    plt.clf()
    plt.plot(winkel[:-1],imp[:-1],'rx',label=r'$Intensität$')
    plt.axvline(winkel_l3,ls='--', color='b',label=r'$L_3 \, \mathrm{Kante}$')
    plt.axvline(winkel_l2,ls='--', color='g',label=r'$L_2 \, \mathrm{Kante}$')
    plt.xlabel(r'$\theta \, \mathrm{in} \, \mathrm{deg}$')
    plt.ylabel(r'$I \, \mathrm{in} \, \mathrm{Imp}/\mathrm{s}$')
    plt.grid()
    plt.xlim(winkel[0]-x_m,winkel[-2]+x_p)
    plt.ylim(min(imp[:-1])-y_m,max(imp[:-1])+y_p)
    plt.legend(loc='best',)
    plt.savefig(name + '.pdf')
    l.Latexdocument(name+'.tex').tabular([winkel[:len(winkel)//2],imp[:len(winkel)//2],winkel[len(winkel)//2:],imp[len(winkel)//2:]],
    '{$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$} & {$\\theta \, / \, \si{\\degree}$} & {$I \, / \, \mathrm{Imp}/\mathrm{s}$}', [2, 0,2,0] ,
    caption = 'Messwerte bei der Untersuchung des Absorptionsspektrums von $\ce{Au}$..', label = name)
    #return energie_k



##Dateneinlesen

winkel_zirkonium, int_zirkonium = np.genfromtxt('zirkonium_ab.txt',unpack=True)
winkel_germanium, int_germanium = np.genfromtxt('germanium_ab.txt',unpack=True)
winkel_zink, int_zink = np.genfromtxt('zink_ab.txt',unpack=True)
winkel_brom, int_brom = np.genfromtxt('brom_ab.txt',unpack=True)
winkel_stom, int_strom = np.genfromtxt('strontium_ab.txt',unpack=True)
winkel_gold, int_gold = np.genfromtxt('gold.txt',unpack=True)

##Plots



##Auswertung
energie_k=[]
energie_k.append(1e3*absorb(12,18,40,0.5*winkel_zirkonium,int_zirkonium,0.5,0.5,10,1,'zr'))
energie_k.append(1e3*absorb(5,10,32,0.5*winkel_germanium,int_germanium,0.5,0.5,1,1,'germanium'))
energie_k.append(1e3*absorb(15,21,35,0.5*winkel_brom,int_brom,0.5,0.5,1,1,'brom'))
energie_k.append(1e3*absorb(14,20,38,0.5*winkel_stom,int_strom,0.25,0.25,10,1,'strom'))
energie_k.append(1e3*absorb_zink(14,20,8,25,30,0.5*winkel_zink,int_zink,0.25,0.25,20,10,'zink'))
absorb_gold(14,20,79,0.5*winkel_gold,int_gold,0.25,0.25,10,1,'gold')



### Bestimmung der Rydbergenergie
#Z=[40-2.88,32-3.09,35-2.98,38-2.98,30-3.37]
Z=[40,32,35,38,30]
Element=['$\ce{Zr}$', '$\ce{Ge}$', '$\ce{Br}$', '$\ce{Sr}$', '$\ce{Zn}$']
l.Latexdocument('ryd.tex').tabular([Z,np.sqrt(energie_k)],
'{Element} & {Z} & {$E_{\mathrm{K}}$}', [ 0,1] ,
caption = 'Experimentell bestimmten Energie $E_{\mathrm{K}}$', label = 'ener_ryd')

def g(m,x,b):
    return m*x+b
parms, cov = curve_fit(g,Z,np.sqrt( energie_k) )
error= np.sqrt(np.diag(cov))
m_u=ufloat(parms[0],error[0])
b_u=ufloat(parms[1],error[1])
print('Regressionsrechnugng E_k und Z')
print('Steigung/Rydberggenerie',m_u)
print('Steigung/Rydberggenerie',m_u**2)
print('y_achsenabschnitt', b_u)
print('\n\n\n')
lauf=np.linspace(28,45,10000)
plt.clf()
plt.xlim(28,42)
#plt.ylim(np.sqrt(8),np.sqrt(20))
plt.plot(Z,np.sqrt(energie_k),'rx',label=r'$\mathrm{Energiewert}$')
plt.plot(lauf,parms[0]*lauf+parms[1], 'b-', label=r'$\mathrm{Regressiongerade}$')
plt.grid()
plt.xlabel(r'$\mathrm{Z}$')
plt.ylabel(r'$\sqrt{E_{\mathrm{K}}} \, \mathrm{in} \, \mathrm{eV}$')
plt.legend(loc='best')
plt.savefig('energie_z.pdf')
