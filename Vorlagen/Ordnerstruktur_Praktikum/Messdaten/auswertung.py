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
data = [f, c, d], #data incl. unpuarray
header = '\multicolumn{2}{c}{$\Delta Q \:/\: \si{\giga\elementarycharge}$} & {$T/ \si{\\micro\second}$} & {$T/ \si{\\micro\second}$}',
places = [(1.1, 1.1), 1, 1],
caption = 'testcaption.',
label = 'testlabel')
##############################################


#r.makeresults()
