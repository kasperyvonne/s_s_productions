import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

fallzeit_kleine_kugel, fallzeit_große_kugel = np.genfromtxt('fallzeit_raumtemp.txt',unpack=True)
temperatur, fallzeit_kugel_t = np.genfromtxt('fallzeit_temperatur.txt',unpack=True)
masse_1, masse_2 = np.genfromtxt('masse.txt',unpack=True)
durchmesser_1,durchmesser_2=np.genfromtxt('durchmesser.txt',unpack=True)

#Umrechnung
temperatur+=273.16
durchmesser_gross=durchmesser_2*1e-3
durchmesser_1*=0.5*1e-3
durchmesser_2*=0.5*1e-3
masse_1*=1e-3
masse_2*=1e-3
print(durchmesser_1)
print(durchmesser_2)
print('\n')
print(masse_1)
print(masse_2)

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


masse_klein=mittel_und_abweichung(masse_1)
masse_gross=mittel_und_abweichung(masse_2)
print('Masse Kugel 1', masse_klein)
print('Masse Kugel 2', masse_gross)
print('\n')

radius_klein=mittel_und_abweichung(durchmesser_1)
radius_gross=mittel_und_abweichung(durchmesser_2)
durchmesser_rohr=mittel_und_abweichung(2*durchmesser_2)
print('Radius Kugel 1', radius_klein)
print('Radius Kugel 2',radius_gross)
print('\n')



def volumen_kugel(radius):
	return 4/3*np.pi*radius**3

def Dichte(volumen, masse):
	return masse/volumen

dichte_klein=Dichte(volumen_kugel(radius_klein),masse_klein)
dichte_gross=Dichte(volumen_kugel(radius_gross),masse_gross)
print('Dichte Kleine Kugel',dichte_klein)
print('dichte Große Kugel',dichte_gross)
print('\n')


#Dichte der Flüssigkeit einfügen:
temp_wasser, dichte_wasser=np.genfromtxt('wasser_dichte_temp_gekuerzt.txt',unpack=True)
dichte_wasser*=1e3
dichte_wasser_20=9.982100000000000417e2
# Berechnung Viskosität und apperaturkonstante
apperaturkonst_klein=0.07640e-6 #(Pam^3/kg)
fallstrecke=float(100e-3)

fallzeit_mittel_klein=mittel_und_abweichung(fallzeit_kleine_kugel)
fallzeit_mittel_gross=mittel_und_abweichung(fallzeit_große_kugel)
fallzeit_mittel_gross_t=mittel_und_abweichung_intervall(fallzeit_kugel_t,2)

print('durchschnittliche Fallzeit - kleine Kugel',fallzeit_mittel_klein)
print('durchschnittliche Fallzeit - grosse Kugel',fallzeit_mittel_gross)
print('\n')
print('Temperatur - durchschnittliche Fallzeit - grosse Kugel',temperatur[::2],fallzeit_mittel_gross_t)
print('\n')

geschwindigkeit_mittel_klein=fallstrecke/fallzeit_mittel_klein
geschwindigkeit_mittel_gross=fallstrecke/fallzeit_mittel_gross

geschwindigkeit_mittel_gross_t=[]
for t in fallzeit_mittel_gross_t:
	geschwindigkeit_mittel_gross_t.append(fallstrecke/t)


print('geschwindigkeit_mittel-kleine Kugel',geschwindigkeit_mittel_klein)
print('durchschnittliche Fallzeit - grosse Kugel',geschwindigkeit_mittel_gross)
print('\n')
print('Temperatur - geschwindigkeit_mittel-grosse Kugel', temperatur[::2],geschwindigkeit_mittel_gross_t)
print('\n')


def viskositaet(dichte_wasser,dichte_kugel,fallzeit_mittel):
	return (apperaturkonst_klein*(dichte_kugel-dichte_wasser)*fallzeit_mittel)

viskositate_kugel_klein=viskositaet(dichte_wasser_20,dichte_klein,fallzeit_mittel_klein)
print
print('viskositate_kugel_klein', viskositate_kugel_klein)
print('\n')

def apperaturkonstante_grose_kugel(dichte_kugel,dichte_wasser,fallzeit_mittel_gross):
	return (viskositate_kugel_klein/((dichte_kugel-dichte_wasser)*fallzeit_mittel_gross))

apperaturkons_grose=apperaturkonstante_grose_kugel(dichte_gross,dichte_wasser_20,fallzeit_mittel_gross)
print('apperaturkons_gross',apperaturkons_grose)
print('\n')


def viskositaet_gross(dichte_wasser,dichte_kugel,fallzeit_mittel):
	return apperaturkons_grose*(dichte_kugel-dichte_wasser)*fallzeit_mittel


viskositat_temperatur=viskositaet_gross(dichte_wasser,dichte_gross,fallzeit_mittel_gross_t)
print('viskositaet_temperatur', temperatur[::2],viskositat_temperatur)
print('\n')

#Reynold_zahl
def reynold_zahl(dichte_wasser,geschwindigkeit_mittel_gross,durchmesser,viskositate_kugel_klein):
	return (dichte_wasser*geschwindigkeit_mittel_gross*durchmesser)/viskositate_kugel_klein 


dichte_wasser_array=np.array([ufloat(n, 0) for n in dichte_wasser])
geschwindigkeit_mittel_gross_array=np.array(geschwindigkeit_mittel_gross_t)
viskositaet_temperatur_array=np.array(viskositat_temperatur)

#print(type(dichte_wasser_array), len(dichte_wasser_array), dichte_wasser_array[0])
#print(type(geschwindigkeit_mittel_gross_array),len(geschwindigkeit_mittel_gross_array), geschwindigkeit_mittel_gross_array[0])
#print(type(viskositaet_temperatur_array),len(viskositaet_temperatur_array), viskositaet_temperatur_array[0])
#print(durchmesser_rohr)
rey_zahl=reynold_zahl(dichte_wasser_array,geschwindigkeit_mittel_gross_array,durchmesser_rohr,viskositaet_temperatur_array)

print('Reynold Zahl',rey_zahl)
print('\n')

##Regressionsrechnung

def f(x,a,b):
	return a*np.exp(b/x)

params,covariance=curve_fit(f,temperatur[::2],unp.nominal_values(viskositat_temperatur))

print(params)
print('\n')

##Plotbereich

plt.xlim(1/300,1/350)
plt.ylim(1e-4,1e-2)
aufvariabele=np.linspace(273.16,350,1000)
plt.plot(1/temperatur[::2] ,unp.nominal_values(viskositat_temperatur),'rx',label='Messwerte')
plt.plot(1/aufvariabele,f(aufvariabele,*params),'b-',label='Regressions Kurve')
plt.yscale('log')
plt.grid(True,which="both")
plt.legend(loc='best')
plt.xlabel(r'$1/T\ in \ \mathrm{K} $')
plt.ylabel(r'$\log{\eta} \ in \ \mathrm{P\!a}\, \mathrm{s}$')
#plt.tight_layout()
#plt.show()
#plt.savefig('viskositaet_temp_log.pdf')