import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Hier nicht wichtig


abstand, schwingungsdauer = np.genfromtxt('eigenmoment.txt',unpack=True)
winkelrecht_passiv_wert,winkelrecht_passiv_fehler =np.genfromtxt('Winkelrichtgröße_passiv.txt',unpack=True)
winkelrecht_passiv=ufloat(winkelrecht_passiv_wert,winkelrecht_passiv_fehler)
schingungsdauer_grauer_zylinder=np.genfromtxt('T_zylinder_gross.txt',unpack=True)
schwingungsdauer_kugel=np.genfromtxt('T_grosse_kugel.txt',unpack=True)
schwingungsdauer_puppe_1=np.genfromtxt('T_puppe_postition_1.txt',unpack=True)
schwingungsdauer_puppe_2=np.genfromtxt('T_puppe_position_2.txt',unpack=True)

#Berechnung von  D pasiv:

#def D(rad, wink, kraf):
#	return (kraf*rad)/wink
#
#wi_rad=[]
#for w in wi:
#	wi_rad.append(math.radians(w))
#print(wi_rad)
#winkelricht=[]
#while n<len(abst):
#	print('Bin dirn')
#	winkelricht.append( D(abst[n],wi_rad[n],kra[n]))
#	n+=1
#print(winkelricht)
#winkelricht_mitt=sum(winkelricht)/len(winkelricht)
#winkelricht_abweichung_mitt= 1/(np.sqrt(len(winkelricht)))*np.std(winkelricht)
#
#winkelrecht_passiv=ufloat(winkelricht_mitt,winkelricht_abweichung_mitt)
#
#print(winkelrecht_passiv)
#
#print(winkelrecht_passiv.n)

#Berehnung von D dynamisch:
hoehe_zylinder=0.5*3.49

abstand=abstand+hoehe_zylinder

abstand*=1e-2
schwingungsdauer*=(1/5)
schwingungsdauer_hilf=[]
schwingungsdauer_abweichung=[]

#print(abstand[::3])
for i in range(len(schwingungsdauer))[::3]:
    schwingungsdauer_hilf.append(np.mean(schwingungsdauer[i:i+3]))
    schwingungsdauer_abweichung.append(1/(np.sqrt(len(schwingungsdauer[i:i+3])))*np.std(schwingungsdauer[i:i+3]))
schwingungsdauer_mittel=np.array(schwingungsdauer_hilf)
schwingungsdauer_u=unp.uarray(schwingungsdauer_hilf,schwingungsdauer_abweichung)
print(schwingungsdauer_u)
#print('Zeitlichesmittel_abweichung ', schwingungsdaer_abweichung_mitt)


def f(m,u,b):
	return m*u+b

params_p,covarian=curve_fit(f,abstand[::3]**2,schwingungsdauer_mittel**2)

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


m,m_e,b,b_e=linregress(abstand[::3]**2,schwingungsdauer_mittel**2)

m_fehler=ufloat(m,m_e)
b_fehler=ufloat(b,b_e)
print('\n')
print('Steigung ', m_fehler)
print('y-Achsenabschnitt', b_fehler)
print('\n')

m_1=222.51e-3
m_2=223.46e-3
winkelrichtgroesse_dynamisch=(((m_1+m_2)*4*np.pi**2)/m_fehler)
print('Winkelrichtgroesse-dynamisch ', winkelrichtgroesse_dynamisch)
print('Winkelrichtgröße_passiv', winkelrecht_passiv)
print('\n')



#Berechnung vom Eigenträgheitsmoment

def J_zylinder(radius,hoehe,masse):
	return (1/4)*masse*radius**2+(1/12)*masse*hoehe**2

radius_zylinder=(3.49*0.5)*1e-2
hoehe_zylinder=(14.05*0.5)*1e-2
J_zylinder1=J_zylinder(radius_zylinder,hoehe_zylinder,m_1)
J_zylinder2=J_zylinder(radius_zylinder,hoehe_zylinder,m_2)

J_eigen=(winkelrichtgroesse_dynamisch)/(4*np.pi**2)*b_fehler-(J_zylinder1+J_zylinder2)

print('Trageheitsmoment_Eigen ', J_eigen)
print('\n')

def traegheitsmoment(schwingungsdauer):
	return (schwingungsdauer**2*winkelrichtgroesse_dynamisch)/(4*(np.pi)**2)

def Mittelwert(schwingungsdauer):
	mittel_schingungsdauer=sum(schwingungsdauer)/len(schwingungsdauer)
	mittel_schingungsdauer_fehler= 1/(np.sqrt(len(schwingungsdauer)))*np.std(schwingungsdauer)
	return ufloat(mittel_schingungsdauer,mittel_schingungsdauer_fehler)

#Berechnung Trägheitsmoment Zylinder grau
schingungsdauer_grauer_zylinder*=(1/5)
schingungsdauer_grauer_zylinder_u=Mittelwert(schingungsdauer_grauer_zylinder)
traeg_zylinder_grau=traegheitsmoment(schingungsdauer_grauer_zylinder_u)
print('Trägheitsmoment_Zylinder grau ', traeg_zylinder_grau)
print('\n')

#Berechnung Trägheitsmoment Kugel

schwingungsdauer_kugel*=(1/5)
schwingungsdauer_kugel_u=Mittelwert(schwingungsdauer_kugel)
traeg_kugel=traegheitsmoment(schwingungsdauer_kugel_u)
print('Trägheitsmoment_Kugel', traeg_kugel)
print('\n')

#Berechnung Trägheitsmoment Puppe Position 1
schwingungsdauer_puppe_1*=(1/5)
schwingungsdauer_puppe1_u=Mittelwert(schwingungsdauer_puppe_1)
traeg_puppe1_u=traegheitsmoment(schwingungsdauer_puppe1_u)
print('Trägheitsmoment_Puppe 1', traeg_puppe1_u)
print('\n')

#Berechnung Trägheitsmoment Puppe Position 2
schwingungsdauer_puppe_2*=(1/5)
schwingungsdauer_puppe2_u=Mittelwert(schwingungsdauer_puppe_2)
traeg_puppe2_u=traegheitsmoment(schwingungsdauer_puppe2_u)
print('Trägheitsmoment_Puppe 2', traeg_puppe2_u)
print('\n')


#Puppe_mass_mitteler
kopf=[3.09e-2,2.39e-2]
kopf_u=Mittelwert(kopf)

arm=[1.62e-2,1.32e-2,1.48e-2]
arm_u=Mittelwert(arm)
arm_laenge=14.00e-2

torso=[3.855e-2,2.58e-2,3.965e-2]
torso_u=Mittelwert(torso)
torso_hoehe=9.77e-2

bein=[1.92e-2,1.60e-2]
bein_u=Mittelwert(bein)
bein_laenge=15.8e-2

def volumen_zylinder(radius,hoehe):
	return np.pi*radius**2*hoehe
def volumen_kugel(radius):
	return (4/3)*np.pi*radius**3

volumen_puppe=volumen_kugel(kopf_u)+2*volumen_zylinder(arm_u,arm_laenge)+volumen_zylinder(torso_u,torso_hoehe)+2*volumen_zylinder(bein_u,bein_laenge)
print('Volumen Puppe', volumen_puppe)
print('\n')

#Theoretische Berechnung der Trägheitsmomente:

def traeg_kugel(masse,radius):
	return (2/5)*masse*radius**2
def trag_zylinder(masse,radius):
	return (1/2)*masse*radius**2
def trag_zylinder_z(masse,radius):
	return (1/12)

masse_kugel=1005.8e-3
radius_kugel=(0.5*13.78)*1e-2

radius_grauerzylinder=(4.00)*1e-2
masse_zylindergrau=812.46e-3

traeg_kugel_theo=traeg_kugel(masse_kugel,radius_kugel)

print('traegheitsmoment_kugel_theo', traeg_kugel_theo)
print('\n')

traegheitsmoment_zylindergrau_theo=trag_zylinder(masse_zylindergrau,radius_grauerzylinder)
print('traegheitsmoment_grauerkegel_theo',traegheitsmoment_zylindergrau_theo)
print('\n')

#Theoretische Berechnung der Trägheitsmomente für die Puppe:

def satz_von_steiner(traegheitsmoment,masse,verschiebung):
	return traegheitsmoment+masse*verschiebung**2

verschiebung_arm=9.37*1e-2
verschiebung_bein_p1=1.19*1e-2
verschiebung_bein_p2=6.77*1e-2

masse_puppe=161.90*1e-3
volumen_bein=volumen_zylinder(bein_u,bein_laenge)
print('Volumen Bein' ,volumen_bein)
volumen_bein_prozentual=volumen_bein/volumen_puppe

volumen_arm=volumen_zylinder(arm_u,arm_laenge)
print('volumen Arm',volumen_arm)
print('\n')
volumen_arm_prozentual=volumen_arm/volumen_puppe

print('Prozentualer-Anteil Bein',volumen_bein_prozentual)
print('prozentualer-Anteil Arm',volumen_arm_prozentual)
print('\n')

masse_arm=masse_puppe*volumen_arm_prozentual
masse_bein=masse_puppe*volumen_bein_prozentual

print('Masse des Arms', masse_arm)
print('Masse des Bein', masse_bein)
print('\n')

volumen_torso=volumen_zylinder(torso_u,torso_hoehe)
volumen_torso_prozentual=volumen_torso/volumen_puppe
masse_torso=masse_puppe*volumen_torso_prozentual


volumen_kopf=volumen_kugel(kopf_u)
volumen_kopf_prozentual=volumen_kopf/volumen_puppe
masse_kopf=masse_puppe*volumen_kopf_prozentual

##Position 1:
#def J_zylinder(radius,hoehe,masse)
trag_puppe_theo_1=2*satz_von_steiner(trag_zylinder(masse_bein,bein_u),masse_bein,verschiebung_bein_p1)
+trag_zylinder(masse_torso,torso_u)+traeg_kugel(masse_kopf,kopf_u)+2*satz_von_steiner(J_zylinder(arm_u,arm_laenge,masse_arm),masse_arm,verschiebung_arm)
print('Theoretisches Trägheitsmoment Puppe  1',trag_puppe_theo_1)

##Postiion 2:
trag_puppe_theo_2=2*satz_von_steiner(trag_zylinder(masse_bein,bein_u),masse_bein,verschiebung_bein_p2)
+trag_zylinder(masse_torso,torso_u)+traeg_kugel(masse_kopf,kopf_u)+2*satz_von_steiner(J_zylinder(arm_u,arm_laenge,masse_arm),masse_arm,verschiebung_arm)
print('Theoretisches Trägheitsmoment Puppe  2',trag_puppe_theo_2)
print('\n')


#Plotbereich
plt.xlim(0,0.07)
#plt.ylim(0,52)
x=np.linspace(0,0.28,1000)

plt.plot(abstand[::3]**2,schwingungsdauer_mittel**2,'rx',label='Messwerte')
plt.plot(x**2,m*x**2+b,'b-',label='Lineare Regression')
#plt.plot(abstand[::3]**2,f(abstand[::3]**2,*params_p),'y-',label='Fit')
plt.legend(loc='best')
plt.ylabel(r'$T^2 \ in \ \frac{1}{\mathrm{s}^2}$')
plt.xlabel(r'$r^2 \ in  \ \mathrm{m}^2$')
plt.grid()
#plt.savefig('lineare_regression.pdf')
#plt.show()


#np.savetxt('schwingungsdauer_dynamisch_gemittelt.txt', np.column_stack([unp.nominal_values(schwingungsdauer_u),unp.std_devs(schwingungsdauer_u)]), header='Schwingungsdauermittel Abweichung')
#np.savetxt('schwingungsdauer_dynamisch_gemi.txt', np.column_stack([winkelrichtgroesse_dynamisch.n,winkelrichtgroesse_dynamisch.s]), header='Winkelrichtgröße Fehler')
#np.savetxt('Schallgeschwindigkeit_Mittelwert.txt',np.column_stack([c_m,c_f]),header='Mittelwert Fehler')
#np.savetxt('Geschwindigkeit.txt',d.T,header='Gang Geschindigkeit Fehler')
#np.savetxt('Inverse der Wellenlaenge.txt',np.column_stack([inwell]),header='Inverse_der_wellelaenge')