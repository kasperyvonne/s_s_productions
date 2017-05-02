import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
import result
#import pandas as pd
r = result.Results()
u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'ohm')
c = Q_(0, 'degC')


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

##Wellenlänge-Verschiebung

weg, anzahl_weg= np.genfromtxt('anzahl_weg.txt', unpack=True)
weg*=1e-3


def weg_lambda(weg,anzahl):
	return (2*weg)/(anzahl)
wellenlänge=weg_lambda(weg,anzahl_weg)
wellenlänge_u=unp.uarray(np.mean(wellenlänge), np.std(wellenlänge)/np.sqrt(len(wellenlänge)) )
print('Wellenlänge', wellenlänge_u)
print(wellenlänge)
print(anzahl_weg)

l.Latexdocument('abstands_messung.tex').tabular([weg*1e3, anzahl_weg, wellenlänge*1000],
'{Weg in $\si{\meter}$} &{Anzahl} & {Wellenlänge in $\si{\\nano\meter}$}', [2, 0, 9] ,
caption = 'Messergebnisse bei der Abstandsmessung', label = 'tab: messwerte_abstand')

##Brechungsindex_luft

druck_luft,anzahl_luft=np.genfromtxt('anzahl_luft.txt',unpack=True)
druck_kohlenstoff, anzahl_kohlenstoff=np.genfromtxt('anzahl_kohlenstoff.txt',unpack=True)
def delta_brechungsindex(anzahl, wellenlaenge):
	return (anzahl*wellenlaenge)/(2*0.05)

def brechungsindex (deltadruck,deltabrechung):
	return 1+deltabrechung*(1.0132*293.15)/(273.15*deltadruck)

delta_brechungsindex_luft=delta_brechungsindex(anzahl_luft,wellenlänge_u)
delta_brechungsindex_kohlenstoff=delta_brechungsindex(anzahl_kohlenstoff,wellenlänge_u)
brechungsindex_luft=brechungsindex(druck_luft,delta_brechungsindex_luft)
brechungsindex_kohlenstoff=brechungsindex(druck_kohlenstoff,delta_brechungsindex_kohlenstoff)

delta_brechungsindex_luft_u=unp.uarray(np.mean(unp.nominal_values(delta_brechungsindex_luft)), np.std(unp.nominal_values(delta_brechungsindex_luft))/unp.sqrt(len(delta_brechungsindex_luft)) )
delta_brechungsindex_kohlenstoff_u=unp.uarray(np.mean(unp.nominal_values(delta_brechungsindex_kohlenstoff)), np.std(unp.nominal_values(delta_brechungsindex_kohlenstoff))/unp.sqrt(len(delta_brechungsindex_kohlenstoff)) )
brechungsindex_luft_u=unp.uarray(np.mean(unp.nominal_values(brechungsindex_luft)), np.std(unp.nominal_values(brechungsindex_luft))/unp.sqrt(len(brechungsindex_luft)) )
brechungsindex_kohlenstoff_u=unp.uarray(np.mean(unp.nominal_values(brechungsindex_kohlenstoff)), np.std(unp.nominal_values(brechungsindex_kohlenstoff))/unp.sqrt(len(brechungsindex_kohlenstoff)) )

l.Latexdocument('brechung_luft.tex').tabular([druck_luft, anzahl_luft, unp.nominal_values(delta_brechungsindex_luft),unp.std_devs(delta_brechungsindex_luft),unp.nominal_values(brechungsindex_luft),unp.std_devs(brechungsindex_luft)],
'{$p-p\'$ in $\si{\\bar}$} &{Anzahl} & {$\Delta n$} & {$\sigma_{\Delta n}$} &{$n$} & {$\sigma_n$}', [1, 0,5,5,5 ,5] ,
caption = 'Messergebnisse für die Brechungszahl bei Luft', label = 'tab: messwerte_luft')

l.Latexdocument('brechung_kohlenstoff.tex').tabular([druck_kohlenstoff, anzahl_kohlenstoff, unp.nominal_values(delta_brechungsindex_kohlenstoff),unp.std_devs(delta_brechungsindex_kohlenstoff),unp.nominal_values(brechungsindex_kohlenstoff),unp.std_devs(brechungsindex_kohlenstoff)],
'{$p-p\'$ in $\si{\\bar}$} &{Anzahl} & {$\Delta n$} & {$\sigma_{\Delta n}$} &{$n$} & {$\sigma_n$}', [1, 0,5,5,5 ,5] ,
caption = 'Messergebnisse für die Brechungszahl bei Luft', label = 'tab: messwerte_luft')


print('Delta brechungsindex_luft', delta_brechungsindex_luft_u)
print('Delta brechungsindex_kohlenstoff', delta_brechungsindex_kohlenstoff_u)
print('brechungsindex_luft', brechungsindex_luft_u)
print('brechungsindex_kohlenstoff', brechungsindex_kohlenstoff_u)







#Plotbereich

#plt.xlim()
#plt.ylim()
#plt.plot(,,'rx',label='')
#
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel('', fontsize=16)
#plt.ylabel('', fontsize=16)
#plt.show()
#plt.savefig('.pdf')


#r.makeresults()
