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

#r.makeresults()

def E_00(I_0, omega, r):
    return I_0*np.exp( (-2*r**2)/omega**2 )

def E_10(I_0, omega, r):
    return I_0* (8*r**2/omega**2)*np.exp( (-2*r**2)/omega**2)

def polarisation( I_0, phi_0, phi):
    return I_0*np.sin(phi-phi_0)**2

def lamb(spaltbreite, n, abstand_haupt, abstand_neben):
    bib={}
    bib['theta']=unp.arctan(abstand_neben/abstand_haupt)
    bib['lambda']=spaltbreite/n*unp.sin(bib['theta'])
    return bib

n, abstand= np.genfromtxt('wellenlaenge.txt', unpack=True)
test=lamb( 1e-5, n, 83.0e-2,(abstand/2)*1e-2)
print(np.mean(test['lambda']))

r,I=np.genfromtxt('T_00.txt',unpack=True)
plt.plot(r,I,'rx')
plt.show()

## Auswertung T_00

## Auswertung T_10

## Auswertung Polarisation

## Auswertung Wellenlänge
