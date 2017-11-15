import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
from scipy.stats import sem
import latex as l
#r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#############################################
#EXAMPLE FOR LATEX TABLE WITH UNP ARRAYS
a = [1.1, 2.2, 3.3]
b = [0.1, 0.2, 0.3]
c = [3, 4, 5]
d = [6, 7, 8] #EXAMPLE ARRAYS

f = unp.uarray(a, b) #EXAMPLE UARRAY

l.Latexdocument('latex_example.tex').tabular(
data = [c, d, f], #Data incl. unpuarray
header = ['\Delta Q / \giga\elementarycharge', 'T_1 / \micro\second', 'T_1 / \micro\second'],
places = [1, 1, (1.1, 1.1)],
caption = 'testcaption.',
label = 'testlabel')
##############################################


## Funktionen ##

def mittelwert_zeit(t_1,t_2,t_3,t_4,t_5): ## Funktion um die Zeitmittelwerte der Messreihe zu bestimmen.
    n=len(t_1)
    mittel=[]
    fehler=[]

    for i in range(n):
        zeit= [ t_1[i], t_2[i], t_3[i], t_4[i], t_5[i] ]
        mittel.append(np.mean(zeit))
        fehler.append(sem(zeit))

    return unp.uarray(mittel,fehler)







###  Drehschieber
#Konstanten
enddruck=ufloat(1e-2, 1e-2*0.2)


p_druck, t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5 = np.genfromtxt('./messdaten/drehshieber/drehschieber_druck.txt', unpack=True)

preasure_druck=unp.uarray(p_druck,p_druck*0.2)

zeiten_dreh_druck_gemittelt=mittelwert_zeit( t_ddruck_1, t_ddruck_2, t_ddruck3, t_ddruck4, t_ddruck5)

lograritmierter_druck=unp.log( (preasure_druck-enddruck )/ (preasure_druck[0]-enddruck) )



## Druckkurve

## Leckkurve


###  Turbo

## Druckkurve

## Leckkurve
