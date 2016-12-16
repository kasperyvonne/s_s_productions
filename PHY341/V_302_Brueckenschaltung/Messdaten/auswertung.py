import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry

u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'meter')
c = Q_(0, 'degC')
c.to('kelvin')
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
teil_a_widerstand_2,teil_a_widerstand_3=np.genfromtxt('weath_brueck_teil_1.txt',unpack=True)
#teil_a_widerstand_2,teil_a_verhaeltniR34=np.linspace('Teila_widerstaende.txt'unpack=True)

#Einheitenzuteilung
#teil_a_widerstand_2=Q_(teil_a_widerstand_2,'ohm')
#teil_a_widerstand_3=Q_(teil_a_widerstand_3,'ohm')
r_g=1000
teil_a_widerstand_4=r_g-teil_a_widerstand_3

#Widerstandberechnung

def wider(R_2,R_3,R_4):
	teil_a_r_3durch4=R_3/R_4
	return R_2*teil_a_r_3durch4

teil_a_widerstand_x=wider(teil_a_widerstand_2,teil_a_widerstand_3,teil_a_widerstand_4)
print(teil_a_widerstand_x)
teil_a_widerstand_x_mittel=mittel_und_abweichung_intervall(teil_a_widerstand_x,3)

#print('Teil a, Widerstand R_x',teil_a_widerstand_x)
print('\n')
print('Teil a, r_2', teil_a_widerstand_2)
print('Teil a, R_3 und r_4', teil_a_widerstand_3,teil_a_widerstand_4)
print('Teil a, Widerstand R_x ',teil_a_widerstand_x)
print('Teil a, Widerstand R_x gemittelt',teil_a_widerstand_x_mittel)
print('\n')


##Teil b)

teil_b_c_2,teil_b_r_3=np.genfromtxt('kapazi_mess_teil_2_a.txt',unpack=True)
teil_b_c_2*=1e-9

##Einheitenzuteilung
#teil_b_c_2=Q_(teil_b_c_2*1e-9,'farad') #Nano Farad
#teil_b_r_3_u=Q_(teil_b_r_3,'ohm')
#
teil_b_widerstand_4=r_g-teil_b_r_3

#Kapazitätsbestimmung und Wiederstand

def capa(c_2,R_3,R_4):
	teil_a_r_3durch4=R_4/R_3
	return c_2*teil_a_r_3durch4

teil_b_capatität_cx_ideal=capa(teil_b_c_2,teil_b_r_3,teil_b_widerstand_4)
teil_b_capatität_cx_ideal_mittel=mittel_und_abweichung_intervall(teil_b_capatität_cx_ideal,3)

print('Teil b, c_2', teil_b_c_2)
print('Teil b, R_3 und r_4', teil_b_r_3,teil_b_widerstand_4)
print('Teil b, Kapazität C_x ',teil_b_capatität_cx_ideal)
print('Teil b, Kapazität Cx Ideal Mittel ',teil_b_capatität_cx_ideal_mittel)
print('\n')

##Teil 2 realer Kondensator
teil_b_c2_real,teil_b_r2_real,teil_b_r3_real=np.genfromtxt('kapazi_mess_real_teil_2_b.txt',unpack=True)
teil_b_c2_real*=1e-9
#teil_b_c2_real=Q_(teil_b_c2_real*1e-9,'farad')
#
#teil_b_r2_real=Q_(teil_b_r2_real,'ohm')
#teil_b_r3_real=Q_(teil_b_r3_real,'ohm')
teil_b_widerstand_4_real=r_g-teil_b_r3_real

teil_b_capatität_cx_real=capa(teil_b_c2_real,teil_b_r3_real,teil_b_widerstand_4_real)
teil_b_capatität_cx_real_mittel=mittel_und_abweichung_intervall(teil_b_capatität_cx_real,3)
teil_b_widerstand_rx_real=wider(teil_b_r2_real,teil_b_r3_real,teil_b_widerstand_4_real)
teil_b_widerstand_cx_real_mittel=mittel_und_abweichung_intervall(teil_b_widerstand_rx_real,3)

print('Teil B, c_2 real', teil_b_c2_real)
print('\n')
print('teil b, r-2 real',teil_b_r2_real)
print('\n')
print('Teil b, R_3 und r_4', teil_b_r3_real, teil_b_widerstand_4_real)
print('\n')
print('Teil b, Widerstand real R_x ',teil_b_widerstand_rx_real)
print('\n')
print('Teil b, Kapazität C_x ',teil_b_capatität_cx_real)
print('\n')
print('Teil b, Widerstand Rx real ',teil_b_widerstand_cx_real_mittel)
print('\n')
print('Teil b, Kapatität Cx real',teil_b_capatität_cx_real_mittel)
print('\n')

##Teil c)
teil_c_indu,teil_c_widerstand_2,teil_c_R3=np.genfromtxt('induktivitätmess_teil_3.txt',unpack=True)
teil_c_indu*=1e-3
#Einheitenzuteilung
#teil_c_indu=Q_(teil_c_indu,'henry')
#teil_c_widerstand_2=Q_(teil_c_widerstand_2,'ohm')
#teil_c_R3=Q_(teil_c_R3,'ohm')
teil_c_r4=r_g-teil_c_R3

#Induktivität und Widerstand

def indu(l_2,R_3,R_4):
	teil_a_r_3durch4=R_3/R_4
	return l_2*teil_a_r_3durch4

teil_c_widerstand_rx=wider(teil_c_widerstand_2,teil_c_R3,teil_c_r4)
teil_c_induktivitaet_lx=indu(teil_c_indu,teil_c_R3,teil_c_r4)
teil_c_widerstand_rx_mittel=mittel_und_abweichung_intervall(teil_c_widerstand_rx,3)
teil_c_induktivitaet_lx_mittel=mittel_und_abweichung_intervall(teil_c_induktivitaet_lx,3)

print('Teil c, Wiederstand 2Indu,',teil_c_widerstand_2)
print('\n')
print('Teil c, R_3', teil_c_R3)
print('\n')
print('Teil c, R_4', teil_c_r4)
print('\n')
print('Teil c, Induktivität in mH', teil_c_indu)
print('\n')
print('Teil c, Widerstand Rx ', teil_c_widerstand_rx)
print('Teil c, Widerstand Rx mittel ', teil_c_widerstand_rx_mittel)
print('\n')
print('Teil c, Indu Lx ',teil_c_induktivitaet_lx)
print('Teil c, Indu lx gemittelt (18, 16)', teil_c_induktivitaet_lx_mittel)
print('\n')



###Teil d)

teil_d_widerstand_2,teil_d_c2,teil_d_widerstand_4,teil_d_widerstand_3=np.genfromtxt('maxwell_bruecke_teil_4.txt',unpack=True)
teil_d_c2*=1e-9

##einheitenbestimmung
#teil_d_widerstand_2=Q_(teil_d_widerstand_2,'ohm')
#teil_d_widerstand_3=Q_(teil_d_widerstand_3_'ohm')
#teil_d_widerstand_4=Q_(teil_d_widerstand_4,'ohm')
#c_4=Q_(c_4*1e-9,'farad') #nano Farad
#
#Induktivitätbestimmung
#
def wider_max(r_2,r_3,r_4):
	return(r_2*r_3)/r_4

#
def indu_max(r_2,r_3,c_4):
	return r_2*r_3*c_4

teil_d_widerstand_rx=wider_max(teil_d_widerstand_2,teil_d_widerstand_3,teil_d_widerstand_4)
teil_d_indu_lx=indu_max(teil_d_widerstand_2,teil_d_widerstand_3,teil_d_c2)
teil_d_widerstand_rx_mittel=mittel_und_abweichung_intervall(teil_d_widerstand_rx,3)
teil_d_indu_lx_mittel=mittel_und_abweichung_intervall(teil_d_indu_lx,3)

print(teil_d_widerstand_2)
print('\n')
print('Teil d), Wiederstand Rx ',teil_d_widerstand_rx)
print('Teil d), Wiederstand Rx  mittel ', teil_d_widerstand_rx_mittel)
print('\n')
print('Teil d), Induktivität Lx ',teil_d_indu_lx)
print('Teil d), Induktivität Lx mittel (16 ,18) ', teil_d_indu_lx_mittel)
print('\n')


###Teil e)
teil_e_frequenz,teil_e_u_br,teil_e_u_s=np.genfromtxt('wien_robison_teil_5.txt',unpack=True)

teil_e_test=teil_e_u_br/(2*np.sqrt(2))
teil_e_u_br*=0.5
teil_e_u_br*=1/(np.sqrt(2))

#print(teil_e_u_br)
#print(teil_e_test)


R=1000
C=993*1e-9

#R=Q_(R,'ohm')
#C=Q_(Q*1-9,'farad') #nano Farard
#teil_e_frequenz=Q_(teil_e_frequenz,'hertz')
#teil_e_u_s=Q_=(teil_e_u_s,'volt')
#teil_e_u_br=Q_(teil_e_u_br,'volt')
#
#print('Einheiten der Spannungen noch überprüfen, momentan: ')
#print(teil_e_u_s)
#print(teil_e_u_br)
#print('\n')
##bestimmung omega_0 und Onega
def freq(R,C):
	return (1/(R*C))*((1/(2*np.pi)))
#
teil_e_omega_0=freq(R,C)
print('Teil e, omega_0 ', teil_e_omega_0)
#
def Omega(frequnz,omega_0):
	return frequnz/omega_0

teil_e_Omega=Omega(teil_e_frequenz,teil_e_omega_0)
print('Teil e, OMEGA ', teil_e_Omega)

##bestimmung u_s/u_e
#
teil_e_quotient_usue=teil_e_u_br/teil_e_u_s #Hier nochmal gucken
print('Teil e, Us/U_e experimentell', teil_e_quotient_usue)

def u_su_e_theo(omega):
	return unp.sqrt(1/9*((omega**2-1)**2/((1-omega**2)**2+9*omega**2))) 
#
lauffvariabele=np.linspace(1e-1,210,100000)

teil_e_quotient_usue_theo=u_su_e_theo(lauffvariabele)
print('Teil e, US/U_e theoretisch', teil_e_quotient_usue_theo)
print('\n')
#
#print('Frequenzen',teil_e_frequenz)
np.savetxt('U_b_u_s_q.txt',(teil_e_frequenz,teil_e_u_br,teil_e_u_s,teil_e_quotient_usue),header='Frequenz Brückenspannung Eingangspannung')
f=np.genfromtxt('U_b_u_s_q.txt',unpack=True)
print(f)
###Teil f)

w_min=min(teil_e_u_br)
print('Minimalspannung',w_min)
teil_e_u_s_mittel=mittel_und_abweichung(teil_e_u_s)
print('Mittelwert Speisspannung',teil_e_u_s_mittel)
klirr=w_min/teil_e_u_s_mittel
print('Klirrfaktor', klirr)
u_2_theo=teil_e_u_br/u_su_e_theo(2)

klirr_theo=teil_e_u_br/teil_e_u_s_mittel
klirr_theo_mittel=mittel_und_abweichung(unp.nominal_values(klirr_theo))
print('Klirrfaktor theoretisch',klirr_theo_mittel)

print('\n')
#
###Einheitenzuweisung
#teil_f_u_br=Q_(teil_f_u_br,'volt')
#teil_f_u_1=Q_(teil_f_u_1,'volt')
#
#print('Einheiten der Spannungen noch überprüfen, momentan: ')
#print(teil_f_u_br)
#print(teil_f_u_1)
#print('\n')
#
##Bestimmung U_2
#
#def u_2(u_br):
#	return u_br/u_su_e_theo(2)
#teil_f_u_2=u_2(teil_f_u_br)
#
#print('Teil f, Oberwelle ', teil_f_u_2)
#
#def klirr(u_1):
#	return teil_f_u_2/u_1
#
#teil_f_klirr=flirr(teil_f_u_1)
#print('Teil f, Klirr', teil_f_klirr)



#Plotbereich
plt.xlim(teil_e_Omega[0]-0.01,teil_e_Omega[-1]+10)
#plt.ylim()
#aufvariabele=np.linsspace()
plt.plot(teil_e_Omega,teil_e_quotient_usue,'rx',label='Messwerte')
plt.plot(lauffvariabele,teil_e_quotient_usue_theo,'b-',label='Theoriekurve')
plt.xscale('log')
#plt.xlabel()
#plt.ylabel()
#
plt.grid()
#plt.legend(loc=9)
plt.legend(loc=[0.13,0.84])

plt.axes([0.58,0.15, 0.3,0.3])
plt.xlim(teil_e_Omega[np.argmin(teil_e_quotient_usue)]-0.05,0.05+teil_e_Omega[np.argmin(teil_e_quotient_usue)])
plt.ylim(-0.1*min(teil_e_quotient_usue),2*min(teil_e_quotient_usue))
plt.plot(teil_e_Omega,teil_e_quotient_usue,'rx',label='Messwerte')
plt.plot(lauffvariabele,teil_e_quotient_usue_theo,'b-',label='Theoriekurve')
plt.grid()
plt.title(r'$\mathrm{Vergrößerung \,  um \,  das \, Minimum}$')
plt.xscale('log')


#plt.show()
#plt.savefig('ub_us.pdf')
print('\n')
print(teil_e_u_s)
