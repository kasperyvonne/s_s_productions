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
durchmesser_1*=0.5*1e-3
durchmesser_2*=0.5*1e-3
masse_1*=1e-3
masse_2*=1e-3

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
dichte_wasser*=1*-3
dichte_wasser_20=9.982100000000000417e-01*1-3
# Berechnung Viskosität und apperaturkonstante
apperaturkonst_klein=0.07640e-3 #(mPam^3/kg)
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
	return apperaturkonst_klein*(dichte_kugel-dichte_wasser)*fallzeit_mittel

viskositate_kugel_klein=viskositaet(dichte_wasser_20,dichte_klein,fallzeit_mittel_klein)

print('viskositate_kugel_klein', viskositate_kugel_klein)
print('\n')

#def apperaturkonstante_grose_kugel(dichte_kugel,dichte_wasser,fallzeit_mittel_gross):
#	return (viskositate_kugel_klein/((dichte_kugel-dichte_wasser)*fallzeit_mittel_gross))
#
#apperaturkons_gross=apperaturkonstante_grose_kugel(dichte_kugel_gross,dichte_wasser,fallzeit_mittel_gross):
#print('apperaturkons_gross',apperaturkons_gross)
#print('\n')
#
#viskositaet_temperatur=mittel_und_abweichung(apperaturkonstante_grose_kugel(dichte_wasser,dichte_kugel_gross,fallzeit_mittel_gross),2)
#print('Viskosität Kugel gross, abhngig von der Temperatur',viskositaet_temperatur)
#
#
#viskositaet_temperatur=mittel_und_abweichung_intervall(viskositaet(dichte_wasser,dichte_kugel_gross))
#
#def reynold_zahl(dichte_wasser,geschwindigkeit_mittel_gross,fallstrecke,viskositate_kugel_klein):
#	return (dichte_wasser*geschwindigkeit_mittel_gross,fallstrecke)/viskositate_kugel_klein 
#
#
##Lineare-regression
#m,m_err,b,b_err=linregress(1/tempertur_gross,np.log(viskositaet_temperatur))
#
#m_u=ufloat(m,m_err)
#b_u=ufloat(b,b_err)
#print('Steigung', m_u)
#print('y-Achsenabschnitt',b_u)
#print('\n')
#
##Plotbereich
#
#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace(1/273.16,1/350,1000)
#
#plt.plot(1/tempertur_gross,np.log(viskositaet_temperatur),'rx',label='Messwerte')
#plt.plot(aufvariabele,m*aufvariabele+b,'b-',label='Lineare Regression')
#plt.yscale('log')
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('.pdf')