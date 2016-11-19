import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

tempertur_klein, fallzeit_klein = np.genfromtxt('fallzeiten_klein.txt,unpack=True')
tempertur_gross, fallzeit_gross = np.genfromtxt('fallzeiten_gross.txt,unpack=True')
#Standartabweichung und Mittelwert 
def mittel_und_abweichung(messreihe):
	mittelwert=sum(messreihe)/len(messreihe)
	abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe)))*np.std(messreihe)
	mittel_und_abweichung=ufloat(mittelwert,abweichung_des_mittelwertes)
	return mittel_und_abweichung

#Standartabweichung und Mittelwert für Messreihe mit Intervallen

def mittel_und_abweichung_intervall(messreihe,intervall_laenge):
	mittelwert_abweichung_liste=[]
	for i in range(len(messreihe))[::intervall_laenge]:
		mittelwert=sum(messreihe[i:i+intervall_laenge])/len(messreihe[i:i+intervall_laenge])
		abweichung_des_mittelwertes=1/(np.sqrt(len(messreihe[i:i+intervall_laenge])))*np.std(messreihe[i:i+intervall_laenge])
		mittelwert_abweichung_liste.append(ufloat(mittelwert,abweichung_des_mittelwertes))
	
	return mittelwert_abweichung_liste


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

#Dichte Bestimmung:

masse_kugel_klein=4.4531e-3

def Dichte(volumen, masse):
	return masse/volumen

dichte_klein=Dichte()
dichte_gross=Dichte()
print('Dichte Kleine Kugel',dichte_klein)
print('dichte Große Kugel'),dichte_gross)
print('\n')


#Dichte der Flüssigkeit einfügen:

dichte_wasser=ufloat(,)

# Berechnung Viskosität und apperaturkonstante
apperaturkonst_klein=0.07640e-3 #(mPam^3/kg)
fallstrecke=100e-3

fallzeit_mittel_klein=mittel_und_abweichung_intervall(fallzeit_klein,10)
fallzeit_mittel_gross=mittel_und_abweichung_intervall(fallzeit_gross,10)

print('Temperatur - durchschnittliche Fallzeit - kleine Kugel',tempertur_klein[::10],fallzeit_mittel_klein)
print('Temperatur - durchschnittliche Fallzeit - grosse Kugel',tempertur_gross[::10],fallzeit_mittel_gross)
print('\n')

geschwindigkeit_mittel_klein=fallstrecke/fallzeit_mittel_klein
geschwindigkeit_mittel_gross=fallstrecke/fallzeit_mittel_gross

print('Temperatur - geschwindigkeit_mittel-kleine Kugel', tempertur[::10],geschwindigkeit_mittel_klein)
print('Temperatur - geschwindigkeit_mittel-grosse Kugel', tempertur[::10],geschwindigkeit_mittel_gross)
print('\n')

def viskositaet(dichte_wasser,dichte_kugel,fallzeit_mittel):
	return apperaturkonst_klein*(dichte_kugel-dichte_wasser)*fallzeit_mittel

viskositate_kugel_klein=mittel_und_abweichung(dichte_wasser,dichte_kugel_klein,fallzeit_mittel)

print('viskositate_kugel_klein', viskositate_kugel_klein)

def apperaturkonstante_grose_kugel(dichte_kugel,dichte_wasser,fallzeit_mittel_gross):
	return (viskositate_kugel_klein/((dichte_kugel-dichte_wasser)*fallzeit_mittel_gross))

apperaturkons_gross=mittel_und_abweichung(apperaturkonstante_grose_kugel(dichte_gross,dichte_wasser,fallzeit_mittel_gross))

print('apperaturkons_gross',apperaturkons_gross)
print('\n')



#Plotbereich

plt.xlim()
plt.ylim()
aufvariabele=np.linsspace()

plt.plot(,,'rx',label='')

plt.grid()
plt.legend(loc='best')
plt.xlabel()
plt.ylabel()
plt.show()
plt.savefig('.pdf')