import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties import correlated_values
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity

r.app(r'S\ua{i}', Q_(ufloat(0.11111, 0.00009), 'meter'))
r.makeresults()
