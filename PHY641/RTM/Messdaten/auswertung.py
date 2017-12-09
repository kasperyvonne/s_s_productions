import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as stds
from uncertainties import correlated_values
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex as l
from uncertainties.umath import *
r = l.Latexdocument('results.tex')
u = UnitRegistry()
Q_ = u.Quantity
import pandas as pd
from pandas import Series, DataFrame
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#FITFUNKTIONEN

def linear(x, a, b):
    return a * x + b

def mid(x):
    return ufloat(np.mean(x), np.std(x, ddof = 1))



#EINLESEN DER BILDER
for i in ['foreward', 'backward']:
    print('Auswertung für ' + i + ':')
    img = mpimg.imread('bilder/pic_01_' + i + '.png')
    plt.clf()
    #x und y Werte der abgelesenen horizontalen und vertikalen Minima
    x_h, y_h = np.genfromtxt('data_01_'+ i + '.txt', unpack = True)* 1000
    x_v, y_v = np.genfromtxt('data_01_b_'+ i + '.txt', unpack = True)* 1000
    plt.plot(x_h, y_h, 'rx')
    plt.plot(x_v, y_v, 'bx')
    params_h, cov_h = curve_fit(linear, x_h, y_h)
    params_v, cov_v = curve_fit(linear, x_v, y_v)
    plot_x_h = np.linspace(x_h[0], x_h[-1], 100)
    plt.plot(plot_x_h, linear(plot_x_h, *params_h), 'r-')
    plot_x_v = np.linspace(x_v[0], x_v[-1], 100)
    plt.plot(plot_x_v, linear(plot_x_v, *params_v), 'b-')
    plt.xlim(0, 2500)
    plt.ylim(2500, 0)
    plt.imshow(img)



    #Berechnungen zu den Gittervektoren
    #Arrays mit Fitparametern und Fehlern
    params_hcor = correlated_values(params_h, cov_h)
    params_vcor = correlated_values(params_v, cov_v)


    #x, y Werte der ersten und letzten punkte
    x_first_h, x_last_h = x_h[0], x_h[-1]
    x_first_v, x_last_v = x_v[0], x_v[-1]

    delta_x_h = x_last_h - x_first_h
    delta_y_h = linear(x_last_h, *params_h) - linear(x_first_h, *params_h)
    mid_x_h = delta_x_h/(len(x_h)- 1)
    mid_y_h = delta_y_h/(len(y_h)- 1)
    print(np.sqrt(mid_x_h**2 +  mid_y_h**2))


    delta_x_v = x_last_v - x_first_v
    delta_y_v = linear(x_last_v, *params_v) - linear(x_first_v, *params_v)
    mid_x_v = delta_x_v/(len(x_v) - 1)
    mid_y_v = delta_y_v/(len(y_v) - 1)
    print(np.sqrt(mid_x_v**2 +  mid_y_v**2))
    theta = np.arccos((delta_x_v * delta_x_h + delta_y_v * delta_y_h) / (np.sqrt(delta_x_v**2 + delta_y_v)*np.sqrt(delta_x_h**2 + delta_y_h)))
    print(180 - theta/np.pi * 180)


    plt.savefig('bilder/fit_01_'+ i + '.pdf')
    #durchschnittliche x und y Abstände
    #d_x_h = np.array([x_h[i]-x_h[i-1] for i in range(1, len(x_h))])
    #d_y_h = np.array([y_h[i]-y_h[i-1] for i in range(1, len(y_h))])
    #d_x_v = np.array([x_v[i]-x_v[i-1] for i in range(1, len(x_v))])
    #d_y_v = np.array([y_v[i]-y_v[i-1] for i in range(1, len(y_v))])

    #Bestimmung der STreckungsfaktoren, Literaturwert a
    a = 0.246 * 1000
    print((len(x_v)-1))
    s_y_2 = (   ((len(x_v)-1)*a)**2 - (delta_x_v**2 * (len(x_h)-1)*a)/((delta_x_h)**2)   )/(delta_y_v**2 - delta_y_h**2 * delta_x_v**2/delta_x_h**2)
    s_y = np.sqrt(s_y_2)

    s_x_2 = (((len(x_h)-1)*a)**2 - s_y_2 * delta_y_h**2 )/(delta_x_h**2)
    s_x = np.sqrt(s_x_2)
    print(np.sqrt((s_x*delta_x_v)**2 + (s_y*delta_y_v)**2)/a)


    #Neuer Winkel
    vec_h = np.array([s_x * delta_x_h, s_y * delta_y_h])
    vec_v = np.array([s_x * delta_x_v, s_y * delta_y_v])
    theta_scaled = np.arccos(np.dot(vec_h, vec_v)/(np.sqrt(vec_v[0]**2 + vec_v[1]**2) *np.sqrt((vec_h*vec_h).sum()) )     )
    theta_scaled = theta_scaled/np.pi*180
    print(180 - theta_scaled)
    plt.plot([x_h[0], x_h[0] + vec_h[0]], [linear(x_h, *params_h)[0], linear(x_h, *params_h)[0] + vec_h[1]], 'g-')
    plt.plot([x_v[0], x_v[0] + vec_v[0]], [linear(x_v, *params_v)[0], linear(x_v, *params_v)[0] + vec_v[1]], 'g-')
    plt.savefig('bilder/fit_01_'+ i + '.pdf')
