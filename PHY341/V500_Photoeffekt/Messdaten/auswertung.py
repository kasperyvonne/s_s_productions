import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
import result
from scipy.constants import *
elektronvolt = e
print(elektronvolt)
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


#Angepasstes Programm

###gelb
u_gelb, i_gelb = np.genfromtxt('gelb.txt', unpack=True)
u_gelb=Q_(u_gelb,'volt')
i_gelb=Q_(i_gelb,' nanoampere')

### Teil c

plt.xlim(-1,20)
#plt.ylim()
plt.plot(u_gelb,i_gelb,'yx',label='Gelbes Licht')

plt.grid()
plt.legend(loc='best')
plt.xlabel(r'$\mathrm{Spannung} \, \mathrm{in} \, \mathrm{V}$')
plt.ylabel(r'$\mathrm{Photostrom}  \, \, \mathrm{in} \, \, \mathrm{nA}$')

plt.axes([0.487	,0.15, 0.32,0.32])
plt.xlim(-0.35,0.01)
plt.ylim(-0.0025,0.16)
plt.plot(u_gelb[20:-1],i_gelb[20:-1],'yx', label=r'$Gelbes Licht - Messpunkt$')
plt.xticks([0, -0.1 , -0.2, -0.3], ['0', '-0.1', '-0.2', '-0.3'])
plt.yticks([0, 0.02 , 0.04 , 0.06,0.08, 0.1, 0.12, 0.14 ], ['0', '0.02', '0.04','0.06', '0.08' , '0.1', '0.12', '0.14'])

plt.title(r'$\mathrm{Vergrößerung \,für \, negative \, Spannungen}$')
plt.grid()
#plt.show()
plt.savefig('gelb.pdf')


### Untersuchung der Bremsspannung gelb
u_gelb_brems=u_gelb[20:-1]
i_gelb_brems=i_gelb[20:-1]

def auswertung(spannungen, stromstaerke,name, plus_x, minus_x, plus_y):
	messwerte={}
	messwerte['I_w']=np.sqrt(stromstaerke)
	def g(m,x,b):
		return m*x+b

	parms, cov = curve_fit(g,spannungen.magnitude, np.sqrt(stromstaerke.magnitude) )
	error= np.sqrt(np.diag(cov))
	m_u=Q_(ufloat(parms[0],error[0]),'ampere / volt')
	b_u=Q_(ufloat(parms[1],error[1]), 'ampere')
	messwerte['Steigung_u']=m_u
	messwerte['y-achse_u']=b_u
	u_grenz=-b_u/m_u
	messwerte['Grenzspannung']=u_grenz
	messwerte['Grenzspannung2']=unp.nominal_values(u_grenz.magnitude)

	plt.clf()
	plt.xlim(spannungen.magnitude[0]-minus_x,spannungen.magnitude[-1]+plus_x)
	plt.ylim(-0.01,np.sqrt(stromstaerke.magnitude[0])+plus_y)
	variabel=np.linspace(spannungen.magnitude[0]-1,spannungen.magnitude[-1]+1,1000)
	plt.plot(spannungen,np.sqrt(stromstaerke.magnitude),'rx',label= r'$I_{\mathrm{p}} \, \mathrm{bei} \,$' + '$\mathrm{'+ name +'}$' +r'$\,  \mathrm{Licht}$')
	plt.plot(variabel,g(parms[0],variabel,parms[1]),'b-',label=r'$\mathrm{Regressionsgerade}$ ')
	plt.grid()
	plt.legend(loc='best')
	plt.xlabel(r'$\mathrm{Bremsspannung} \, \mathrm{in} \, \mathrm{V}$')
	plt.ylabel(r'$\sqrt{I_{\mathrm{p}}} \, \, \mathrm{in} \, \, \sqrt{\mathrm{nA}}$')
	#plt.show()
	plt.savefig( name +'.pdf')
	return messwerte


###Gelb

result_gelb=auswertung((-1)*u_gelb_brems,i_gelb_brems,'gelbem',0.08,0.01,0.1)

l.Latexdocument('gelb.tex').tabular([(-1)*u_gelb_brems.magnitude, i_gelb_brems.magnitude, np.sqrt(i_gelb_brems.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}', [3, 3, 3] ,
caption = 'Gemessener Photostrom bei gelbem Licht', label = 'gelb')

l.Latexdocument('gelb_all.tex').tabular([u_gelb.magnitude, i_gelb.magnitude],
'{Spanung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$}', [3, 3] ,
caption = 'Gemessener Photostrom bei gelbem Licht', label = 'gelb_all')

### grün
u_gruen, i_gruen = np.genfromtxt('grün.txt', unpack=True)
u_gruen=Q_(u_gruen,'volt')
i_gruen=Q_(i_gruen,' nanoampere')
result_gruen=auswertung(u_gruen,i_gruen,'grünem',  0.05,0.01,0.1)

l.Latexdocument('grün.tex').tabular([u_gruen.magnitude, i_gruen.magnitude, np.sqrt(i_gruen.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}', [2, 2, 2] ,
caption = 'Gemessener Photostrom bei grünem licht', label = 'gruen')
#'{Bremsspannung $U$ in $\si{\volt}$} & {Photostrom $I\ua{p}$ in $\si{\nano\ampere}$} & {Photostrom $\sqrt{I\ua{p}}$ in $\sqrt{\si{\nano\ampere}}$}', [3, 2, 2] ,




### grün,blau

u_gb, i_gb = np.genfromtxt('grün_blau.txt', unpack=True)
u_gb=Q_(u_gb,'volt')
i_gb=Q_(i_gb,' nanoampere')
result_gruen_blau=auswertung(u_gb,i_gb,'grün-blauem', 0.06,0.01,0.01)

l.Latexdocument('grün_blau.tex').tabular([u_gb.magnitude, i_gb.magnitude, np.sqrt(i_gb.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}', [3, 3, 3] ,
caption = 'Gemessener Photostrom bei grün-blauem licht', label = 'gb')
#'{Bremsspannung $U$ in $\si{\volt}$} & {Photostrom $I\ua{p}$ in $\si{\nano\ampere}$} & {Photostrom $\sqrt{I\ua{p}}$ in $\sqrt{\si{\nano\ampere}}$}', [3, 2, 2] ,

### violett

u_violett, i_violett = np.genfromtxt('violett.txt', unpack=True)
u_violett=Q_(u_violett,'volt')
i_violett=Q_(i_violett,' nanoampere')
result_violett=auswertung(u_violett,i_violett,'violettem', 0.05,0.1,0.1)

l.Latexdocument('violett.tex').tabular([u_violett.magnitude, i_violett.magnitude, np.sqrt(i_violett.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}', [3, 2, 2] ,
caption = 'Gemessener Photostrom bei violettem licht', label = 'violett')
#'{Bremsspannung $U$ in $\si{\volt}$} & {Photostrom $I\ua{p}$ in $\si{\nano\ampere}$} & {Photostrom $\sqrt{I\ua{p}}$ in $\sqrt{\si{\nano\ampere}}$}', [3, 2, 2] ,


##uv  1
u_uv1, i_uv1 = np.genfromtxt('uv_1.txt', unpack=True)
u_uv1=Q_(u_uv1,'volt')
i_uv1=Q_(i_uv1,' nanoampere')
result_uv=auswertung(u_uv1,i_uv1,' UV 1 ', 0.2,0.1,0.1)

l.Latexdocument('uv_1.tex').tabular([u_uv1.magnitude, i_uv1.magnitude, np.sqrt(i_uv1.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}', [3, 2, 2] ,
caption = 'Gemessener Photostrom beim ersten ultravioletten licht', label = 'uv_eins')
#'{Bremsspannung $U$ in $\si{\volt}$} & {Photostrom $I\ua{p}$ in $\si{\nano\ampere}$} & {Photostrom $\sqrt{I\ua{p}}$ in $\sqrt{\si{\nano\ampere}}$}', [3, 2, 2] ,


##uv  2
u_uv2, i_uv2 = np.genfromtxt('uv_2.txt', unpack=True)
u_uv2=Q_(u_uv2,'volt')
i_uv2=Q_(i_uv2,' nanoampere')
result_uv_2=auswertung(u_uv2,i_uv2,' UV 2 ', 0.2,0.1,0.1)
l.Latexdocument('uv_2.tex').tabular([u_uv2.magnitude, i_uv2.magnitude, np.sqrt(i_uv2.magnitude)],
'{Bremsspannung $U$ in $\si{\\volt}$} & {Photostrom $I_{\map{p}}$ in $\si{\\nano\\ampere}$} & {Photostrom $\sqrt{I_{\map{p}}}$ in $\sqrt{\si{\\nano\\ampere}}$}' 	, [3, 2, 2] ,
caption = 'Gemessener Photostrom beim zweiten ultravioletten licht', label = 'uv_zwei')
#'{Bremsspannung $U$ in $\si{\volt}$} & {Photostrom $I\ua{p}$ in $\si{\nano\ampere}$} & {Photostrom $\sqrt{I\ua{p}}$ in $\sqrt{\si{\nano\ampere}}$}', [3, 2, 2]



####Tabelle mit den Messergebnissen

steigungen=[float(unp.nominal_values(result_gelb['Steigung_u'].magnitude)),float(unp.nominal_values(result_gruen['Steigung_u'].magnitude)),float(unp.nominal_values(result_gruen_blau['Steigung_u'].magnitude)),float(unp.nominal_values(result_violett['Steigung_u'].magnitude)),float(unp.nominal_values(result_gelb['Steigung_u'].magnitude)),float(unp.nominal_values(result_uv_2['Steigung_u'].magnitude))]
steigungen_fehler=[float(unp.std_devs(result_gelb['Steigung_u'].magnitude)),float(unp.std_devs(result_gruen['Steigung_u'].magnitude)),float(unp.std_devs(result_gruen_blau['Steigung_u'].magnitude)),float(unp.std_devs(result_violett['Steigung_u'].magnitude)),float(unp.std_devs(result_gelb['Steigung_u'].magnitude)),float(unp.std_devs(result_uv_2['Steigung_u'].magnitude))]


y_abschnitt=[float(unp.nominal_values(result_gelb['y-achse_u'].magnitude)),float(unp.nominal_values(result_gruen['y-achse_u'].magnitude)),float(unp.nominal_values(result_gruen_blau['y-achse_u'].magnitude)),float(unp.nominal_values(result_violett['y-achse_u'].magnitude)),float(unp.nominal_values(result_gelb['y-achse_u'].magnitude)),float(unp.nominal_values(result_uv_2['y-achse_u'].magnitude))]
y_abschnitt_fehler=[float(unp.std_devs(result_gelb['y-achse_u'].magnitude)),float(unp.std_devs(result_gruen['y-achse_u'].magnitude)),float(unp.std_devs(result_gruen_blau['y-achse_u'].magnitude)),float(unp.std_devs(result_violett['y-achse_u'].magnitude)),float(unp.std_devs(result_gelb['y-achse_u'].magnitude)),float(unp.std_devs(result_uv_2['y-achse_u'].magnitude))]

spannungen=[float(result_gelb['Grenzspannung2']),float(result_gruen['Grenzspannung2']),float(result_gruen_blau['Grenzspannung2']),float(result_violett['Grenzspannung2']),float(result_uv['Grenzspannung2']),float(result_uv_2['Grenzspannung2'])]
spannungen_fehler=[float(unp.std_devs(result_gelb['Grenzspannung'].magnitude)),float(unp.std_devs(result_gruen['Grenzspannung'].magnitude)),float(unp.std_devs(result_gruen_blau['Grenzspannung'].magnitude)),float(unp.std_devs(result_violett['Grenzspannung'].magnitude)),float(unp.std_devs(result_gelb['Grenzspannung'].magnitude)),float(unp.std_devs(result_uv_2['Grenzspannung'].magnitude))]


l.Latexdocument('ergebnisse_kompakt.tex').tabular([steigungen, steigungen_fehler,y_abschnitt,y_abschnitt_fehler,spannungen,spannungen_fehler],
'{Farbe} & {$m$ in $\si{\\ampere\per\\volt}$} & { $\sigma_{\map{m}}$ in $\si{\\ampere\per\\volt}$ } & { $b$ in $\si{\\ampere}$} & { $\sigma_{\map{b}}$ in $\si{\\ampere}$} & {$U_{\map{g}}$ in $\si{\\volt}$ } & { $\sigma_{U_{\map{g}}}$ in $\si{\\volt}$ }   ', [2, 2,3 ,3,2,2] ,
caption = 'Messergebnisse für die verschiedenen Wellenlängen', label = 'messergebnisse')




## wellenlänge Gegenspannung

wellenlaenge=np.array([577*1e-9,546*1e-9,492*1e-9,434*1e-9,365*1e-9,366*1e-9])
c=float(299792458)
frequenz=c/wellenlaenge
spannungen=[result_gelb['Grenzspannung2'],result_gruen['Grenzspannung2'],result_gruen_blau['Grenzspannung2'],result_violett['Grenzspannung2'],result_uv['Grenzspannung2'],result_uv_2['Grenzspannung2']]

l.Latexdocument('wellen.tex').tabular([wellenlaenge*1e9, frequenz*1e-12],
'{Farbe} & {$\lambda$ in \si{\\nano\\meter}} & { $f$ in $\si{\THz}$}', [1,1] ,
caption = 'Untersuchtes Lichtspektrum der $\map{HG}$-Lampe ', label = 'wellen')


def g(m,x,b):
	return m*x+b

parms, cov = curve_fit(g,frequenz*1e-12,spannungen)
error= np.sqrt(np.diag(cov))
m_u=Q_(ufloat(parms[0],error[0]),'joule second / coulomb')
b_u=Q_(ufloat(parms[1],error[1]), 'volt')
print('h durch e', m_u)
print('Austrittsarbeit', b_u)


plt.clf()
plt.grid()
lauf=np.linspace(frequenz[0]-1e14,frequenz[-1]+1.5e14,1000)
plt.xlim(0.5*1e15*1e-12,0.85*1e15*1e-12)
plt.ylim(0,2)
plt.plot(frequenz*1e-12,spannungen,'rx', label=r'$\mathrm{Bestimmte} \, \mathrm{Gegenspannung}}$')
plt.plot(lauf*1e-12,g(parms[0],lauf*1e-12,parms[1]),'b-', label=r'$\mathrm{Regressionsgerade}$')
plt.legend(loc='best')
plt.xlabel(r'$\mathrm{f}\,\mathrm{in} \, \mathrm{THz}$')
plt.ylabel(r'$U_{\mathrm{G}} \, \mathrm{in} \, \mathrm{V}$')
#plt.show()
plt.savefig('wellenlaenge_gegen.pdf')

#print(spannungen)

#plt.clf()
#plt.plot(frequenz,spannungen)
#plt.show()
#Plotbereich

#plt.xlim()
#plt.ylim()
#plt.plot(u_gelb,i_gelb,'rx',label='')
#
#plt.grid()x
#plt.legend(loc='best')
#plt.xlabel('', fontsize=16)
#plt.ylabel('', fontsize=16)
#plt.show()
#plt.savefig('.pdf')


#r.makeresults()
