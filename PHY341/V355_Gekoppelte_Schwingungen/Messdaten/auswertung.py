import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex

u = UnitRegistry()
Q_ = u.Quantity

# umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin
# beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5, 4, 3], [0.1, 0.2, 0.3]), 'ohm')
c = Q_(0, 'degC')
c.to('kelvin')
#print(c.to('kelvin'))
#print(a**2)
#print(b**2)
#einheitentst=Q_(1*1e-3,'farad')
#einheitentst_2=Q_(1,'ohm')
#print(einheitentst)
#print(1/(einheitentst*einheitentst_2).to('second'))


#variabel_1,variabel_2=np.genfromtxt('name.txt',unpack=True)

#Standartabweichung und Mittelwert

def mittel_und_abweichung(messreihe):
    messreihe_einheit=messreihe.units
    mittelwert=sum(messreihe)/len(messreihe)
    abweichung_des_mittelwertes=1/((len(messreihe))**0.5)*np.std(messreihe)
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
##Versuchskonstanten
c=Q_(ufloat(0.7932 ,0),' nanofarad')
c_sp=Q_(ufloat(0.028,0),'nanofarad')
l=Q_(ufloat(23.954,0),'millihenry')

##

def v_min (l,c,c_k):
	return 1/(2*np.pi*(l*(1/c+2/c_k)**(-1))**0.5)
def v_plu (l,c):
	return 1/(2*np.pi*(l*c)**0.5)
##Aufgabenteil a

c_k,n=np.genfromtxt('teilaufgabe_a_schwingungsmaxima.txt',unpack=True)
c_k=Q_(unp.uarray(c_k,c_k*0.2),'nanofarad')
n=unp.uarray(n,0.5)
print(type(n))
#Verhältnis
verhaeltnis=1/n
print('Verhaeltnis Schwebung zu Schwingung', verhaeltnis)
print('\n')

#latex.Latexdocument('teila_ck_n.tex').tabular([	c_k.magnitude,n,verhaeltnis],'{$C_k$ in $\si{\\nano\\farad}$$} & {Anzahl der Schwingungsmaxima} & {Verhältnis}',[1,1,1],
#	caption=' Anzahl der Schwingungsmaxima bei verschiedenenen Kapazitäten $C_k$', label='teila_n_ck')


#Bestimmung der Schwingungsfrequenzen
v_mint=v_min(l,c,c_k).to('kilohertz')
v_plut=v_plu(l,c).to('kilohertz')
print('Schwingungsfrequenz Theorie v_-',v_mint)
print('\n')
print('Schwingungsfrequenz Theorie v_+',v_plut)
print('\n')



verhaeltnis=(1/n)
#n_theo=(v_plut+v_mint)/(2*(v_mint-v_plut))
n_theo=(2*(v_mint-v_plut))/(v_plut+v_mint)

print('Theoretisch Verhältnis',n_theo)
n_verhl=((verhaeltnis/(n_theo))-1)*100

latex.Latexdocument('teila_ck_n_mit_theo.tex').tabular([c_k.magnitude,n,verhaeltnis,n_theo,n_verhl],r'{$C\ua{k}$ in $\si{\nano\farad}$} & {Anzahl Schwingungsmaxima} & {Verhäl. $n$ in \%} & { Verhäl. $n\ua{theo}$ in \%} & {rel. Abw. $\frac{n}{n\ua{theo}}$ in \%}' ,[1,1,2,2,1], caption=r'Anzahl der Schwingungsmaxima bei verschiedenenen Kapazitäten $C_k$', label='teila_n_ck')
verhaeltnis=((1/n)-1)*100

#for n in range(len(v_mint)):
#	a[n]=v_plut.magnitude.noms
#	b[n]=v_plu.magnitude.stds
#v_plut_list=unp.uarray(a,b)
#print(v_plut_list)

##Aufgabenteil b
c_k_1,v_plug,v_ming=np.genfromtxt('teilaufgabe_b_frequenzen.txt',unpack=True)
v_ming=Q_(unp.uarray(v_ming,1),'kilohertz')
v_plug=Q_(unp.uarray(v_plug,1),'kilohertz')

##Verhältniss Theorie und Praxis
v_min_verhael=(v_ming/v_mint-1)*100
v_plu_verhael=(v_plug/v_plut-1)*100

print('Verhältnis von v_m',v_min_verhael)
print('Verhältnis von v_+',v_plu_verhael)
print('\n')

latex.Latexdocument('teilb_schwingungen_prak_gemessen_frequenzen.tex').tabular([c_k.magnitude,v_ming.magnitude,v_plug.magnitude,v_min_verhael.magnitude,v_plu_verhael.magnitude],
	r'$C\ua{k}$ in $\si{\nano\farad}$} & {Frequenz $\nu_-$ in $\si{\kilo\hertz}$} & {Frequenz $\nu_+$ in $\si{\kilo\hertz}$}& {rel. Verhältnis $n_-$ in \%} & {rel. Verhältnis $n_+$ in \%}',[1,1,1,1,1],
	caption=' Gemessene Fundamentalfrequenzen bei einer erzwungenen Schwingungen und das relative Verhältnis zu den Theoriewerten', label='teilb_schwingungen_prak_theo')

latex.Latexdocument('teilb_schwingungen_prak_theo_frequenzen.tex').tabular([c_k.magnitude,v_mint.magnitude],r'{$C\ua{k} in $\si{\\nano\\farad}$} &{$\\nu_{-\,\mathup{theo}}$ in $\si{\kilo\hertz}$}',[1,1],
	caption='Theoretisch bestimmte Fundamentalfrequenzen', label='teilb_frequenzen_theo')




v_plu_verhael_mittel=(sum(v_plu_verhael)/len(v_plu_verhael))
print('Gemittelte Abweichung v_plu',v_plu_verhael_mittel)
print('\n')

##Aufgabenteil c

periode=Q_(1,'second')
startf=Q_(15.67,'kilohertz')
endf=Q_(96.15,'kilohertz')
m=(endf-startf)/periode
print('Steigung m',m)
print('\n')

def zeit_f_gerade(t):
	return m*t+startf

c_k,t_1,t_2=np.genfromtxt('teilaufgabe_c_deltaT.txt',unpack=True)
t_1=Q_(unp.uarray(t_1,5),'millisecond')
t_2=Q_(unp.uarray(t_2,5),'millisecond')
c_k=Q_(unp.uarray(c_k,c_k*0.2),'nanofarad')

latex.Latexdocument('teilc_gemessene_zeitabstaende.tex').tabular([c_k.magnitude,t_2.magnitude,t_1.magnitude],
	'{$C\\ua{k} in $\si{\\nano\\farad}$} & {Abstand $\\Delta t_+$ in \si{\milli\second}} & {Abstand $\\Delta t_-$ in \si{\milli\second}}',
	[1,1,1], caption='Gemessene Zeitabstände bei unterschiedlichen $C\\ua{k}$', label='teilc_gemessene_zeit')

frequenzen_t1=zeit_f_gerade(t_1).to('kilohertz')
frequenzen_t2=zeit_f_gerade(t_2).to('kilohertz')
print('Frequenz Teil c v+',frequenzen_t1)
print('\n')
v_pu_verhael_c=(frequenzen_t1/v_plut-1)*100
v_plu_verhael_mittel_c=((sum(v_pu_verhael_c)/len(v_pu_verhael_c))-1)*100

print('Verhältnis Teilc v+, gemittelt', v_pu_verhael_c,v_plu_verhael_mittel_c)
print('\n')
print('Frequenz Teil c v-',frequenzen_t2)
print('\n')
v_min_verhael_c=(frequenzen_t2/v_mint-1)*100
print('Verhältnis Teilc v-', v_min_verhael_c)
print('\n')

latex.Latexdocument('teilc_schwingungen_prak_gemessen_frequenzen.tex').tabular([c_k.magnitude,frequenzen_t2.magnitude,frequenzen_t1.magnitude,v_min_verhael_c.magnitude,v_pu_verhael_c.magnitude],
	'{$C\\ua{k}$ in $\\si{\\nano\\farad}$} & {Frequenz $\\nu_-$ in $\\si{\\kilo\\hertz}$} & {Frequenz $\\nu_+$ in $\\si{\\kilo\\hertz}$ }& {rel. Verhältnis $n_{-}$ in \%} & {rel. Verhältnis $n_+$ in \%}',[1,1,1,1,1],
    caption='Bestimmung der Fundamentalfrequenzen mit der Sweep-Methode und zusätzlich das relatives Verhältnis zu den Theoriewerten.', label='teilc_schwingungen_prak_theo')




#Plotbereich

plt.xlim(noms(c_k[0].magnitude)-0.5,noms(c_k[-1].magnitude)+0.5)
#plt.ylim()
#aufvariabele=np.linsspace()
#
v_plutp=[]
for n in range(len(c_k.magnitude)):
	v_plutp.append(noms(v_plut.magnitude))

#v_mint=v_min(l,c,c_k).to('kilohertz')
#v_plut=v_plu(l,c).to('kilohertz')
c_kl=np.linspace(0.5,c_k[-1].magnitude+1,1000)
#print('Liste C',c_kl)
print(len(c_kl))
v_theo_min_list=v_min((l.magnitude)*1e-3,(c.magnitude)*1e-9,c_kl*1e-9)*1e-3
v_tho_plu_list=v_plu((l.magnitude)*1e-3,(c.magnitude)*1e-9)*1e-3
v_plarray=noms(v_plut.magnitude)*np.ones(len(c_kl))
print(v_plarray)

plt.plot(noms(c_k.magnitude),noms(v_ming.magnitude),'rx',label=r'$Erzwungene \, Schwingung \, \nu_-$')
plt.plot(noms(c_k.magnitude),noms(v_plug.magnitude),'gx',label=r'$Erzwungene \, Schwingung \, \nu_+$')
plt.plot(noms(c_k.magnitude),noms(frequenzen_t2.magnitude),'yx',label='$Sweep-Methode \\, \\nu_-$')
plt.plot(noms(c_k.magnitude),noms(frequenzen_t1.magnitude),'bx',label=r'$Sweep-Methode\, \nu_+$')
plt.plot(noms(c_kl),noms(v_theo_min_list),'c-',label=r'$Theoriekurve \, \nu_-$')
plt.plot(noms(c_kl),v_plarray,'m-',label=r'$Theoriekurve \, \nu_+$')



plt.grid()
plt.legend(loc='best')
plt.xlabel('$C_{\\mathrm{k}} \\, \\, \\mathrm{in} \\,\\, \\mathrm{nF}$')
plt.ylabel('$\\nu \\, \\, \\mathrm{in} \\, \\,  \\mathrm{kHz}$')
#plt.show()
plt.savefig('plot_frequenzen.pdf')
