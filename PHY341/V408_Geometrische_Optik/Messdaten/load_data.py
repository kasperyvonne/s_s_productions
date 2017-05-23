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
r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
def error_s(a):
    return unp.uarray(a, 0.1)


G = 3#Gegenstandsgröße


####METHODE 1
method_1_g_raw, method_1_b_raw, method_1_B_raw = np.genfromtxt('data/nachgemessen.txt', unpack=True)

sort = np.argsort(method_1_g_raw)
method_1_g = method_1_g_raw[sort]
method_1_b = method_1_b_raw[sort]
method_1_B = method_1_B_raw[sort]



######METHODE NACH BESSEL
bessel_g_1_raw, bessel_b_1_raw, bessel_g_2_raw, bessel_b_2_raw = np.genfromtxt('data/bessel.txt', unpack=True)
bessel_g_1 = bessel_g_1_raw
bessel_b_1 = bessel_b_1_raw
bessel_g_2 = bessel_g_2_raw
bessel_b_2 = bessel_b_2_raw
bessel_g = np.concatenate((bessel_g_1, bessel_b_2))
bessel_b = np.concatenate((bessel_b_1, bessel_g_2))



g_1_blau_raw, b_1_blau_raw, g_2_blau_raw, b_2_blau_raw = np.genfromtxt('data/blau.txt', unpack=True)
g_1_blau = g_1_blau_raw
b_1_blau = b_1_blau_raw
g_2_blau = g_2_blau_raw
b_2_blau = b_2_blau_raw
blau_g = np.concatenate((g_1_blau, b_2_blau))
blau_b = np.concatenate((b_1_blau, g_2_blau))

g_1_rot_raw, b_1_rot_raw, g_2_rot_raw, b_2_rot_raw = np.genfromtxt('data/rot.txt', unpack=True)
g_1_rot = g_1_rot_raw
b_1_rot = b_1_rot_raw
g_2_rot = g_2_rot_raw
b_2_rot = b_2_rot_raw
rot_g = np.concatenate((g_1_rot, b_2_rot))
rot_b = np.concatenate((b_1_rot, g_2_rot))



#METHODE NACH ABBE
abbe_g_raw, abbe_b_raw, abbe_B_raw = np.genfromtxt('data/abbe.txt', unpack=True)
abbe_g = abbe_g_raw
abbe_b = abbe_b_raw
abbe_B = abbe_B_raw
abbe_V = abbe_B/G



#wasserlinse
wasser_g_raw, wasser_b_raw = np.genfromtxt('data/wasserlinse.txt', unpack=True)
wasser_g = wasser_g_raw
wasser_b = wasser_b_raw
