import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#from pint import UnitRegistry
#
#u = UnitRegistry()
#Q_ = u.Quantity
#
##umrechnung einheiten mit var.to('unit')
## Einheiten für pint:dimensionless, meter, second, degC, kelvin
##beispiel:
#a = ufloat(5, 2) * u.meter
#b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'meter')
#c = Q_(0, 'degC')
#c.to('kelvin')
#print(c.to('kelvin'))
#print(a**2)
#print(b**2)

#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

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


###Angepasstes Programm 

##Teil a)
#teil_a_widerstand_2,teil_a_widerstand_3,teil_a_widerstand_4=np.linspace('Teila_widerstaende.txt'unpack=True)
#teil_a_widerstand_2,teil_a_verhaeltniR34=np.linspace('Teila_widerstaende.txt'unpack=True)

#Widerstandberechnung

def wider(R_2,R_3,R_4):
	teil_a_r_3durch4=R_3/R_4
	
	u_teil_a_r_3durchr_4=unp.uarray(teil_a_r_3durch4,teil_a_r_3durch4*0.05)
	
	return R_2*u_teil_a_r_3durchr_4

def wider_ver(R_2,R_3dR_4):
	u_teil_a_r_3durchr_4=unp.uarray(R_3dR_4,R_3dR_4*0.05)
	return R_2*u_teil_a_r_3durchr_4

teil_a_widerstand_x=wider(teil_a_widerstand_2,teil_a_widerstand_3,teil_a_widerstand_4)
teil_a_widersta_x_v=wider_ver(teil_a_widerstand_2,teil_a_verhaeltniR34)

#print('Teil a, Widerstand R_x',teil_a_widerstand_x)
print('Teil a, Widerstand R_x',teil_a_widersta_x_v)
print('\n')

##Teil b)
#teil_b_widerstand_2,teil_b_verhaeltniR34=np.linspace('Teila_widerstaende.txt'unpack=True)

teil_b_verhaeltniR34_u=unp.uarray(teil_a_verhaeltniR34,0.05*teil_a_verhaeltniR34)
teil_b_widerstand_2_u=unp.uarray(teil_b_widerstand_2_u,0.03*teil_b_widerstand_2_u)

#Kapazitätsbestimmung

def capa(c_2,R_3dR_4):
	u_teil_a_r_3durchr_4=unp.uarray(R_3dR_4,R_3dR_4*0.05)
	return c_2*R_3dR_4

teil_b_widerstand_rx=wider_ver(teil_b_widerstand_2_u,teil_b_verhaeltniR34)
teil_b_capatität_cx=capa(c_2,teil_b_verhaeltniR34)

print('Teil b, Widerstand Rx',teil_b_widerstand_rx)
print('Teil b, Widerstand Cx',teil_b_capatität_cx)
print('\n')

##Teil c)

#Induktivität







#Plotbereich

#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()
#
#plt.plot(,,'rx',label='')
#
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('.pdf')
