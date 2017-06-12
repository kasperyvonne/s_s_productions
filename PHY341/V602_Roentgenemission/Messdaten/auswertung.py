import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math as m
from scipy.constants import *
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
#r.makeresults()

fine_structure=0.0072973525664
Ryd=13.605693009


###### Versuchvorberitung

def brag_wink (E,n):
    lam= c * h / ((E*e)) # eventuell noch umrechnen
    winkel= np.degrees(np.arcsin( (n*lam) / (2*201.4*1e-12) ))
    return winkel
def brag_ener (winkel, n):
    lam= 2*201.4e-12 * np.sin( m.radians(winkel) ) / n #Eventuell rad to deg
    return h* c/ (lam*e)

def abschirm(Z, delta_E):
    return Z - ( np.sqrt( (4 /fine_structure ) * np.sqrt( delta_E / Ryd) -  ( 5 * delta_E ) / Ryd ) ) * ( np.sqrt( 1 + (19/32) * ( fine_structure )**2 * ( delta_E / Ryd ) ) )



energien=np.array([9.65,10.37,13.48,15.21,16.12,18.00,19.00])*1e3 ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)
Z= np.array([30,32,35,37,38,40,41])  ## (Zink, Germanium, Brom, Rubiidium, Stotium, Niobium)

print('\n')
print(' Abschirmung', abschirm(Z,energien))
print('\n')
print('Bragg Winkel',brag_wink(energien,1))
print('\n \n \n')
