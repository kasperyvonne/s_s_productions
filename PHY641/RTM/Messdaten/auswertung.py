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
#series = pd.Series(data, index=index)
# d = pd.DataFrame({'colomn': series})

#FITFUNKTIONEN

def linear(x, a, b):
    return a * x + b

def inverselinear(y, a, b):
    return (y - b) / a

def mid(x):
    return ufloat(np.mean(x), np.std(x, ddof = 1))
def norm(vec):
    return sqrt(vec[0]**2 + vec[1]**2)
def angle(a, b):
    return acos((np.dot(a, b)) / (sqrt(a[0]**2 + a[1]**2)*sqrt(b[0]**2 + b[1]**2)))/np.pi * 180

globvec = []
#EINLESEN DER BILDER
for i in ['foreward', 'backward']:
    print('Auswertung für ' + i + ':')
    img = mpimg.imread('bilder/pic_01_' + i + '.png')
    plt.clf()
    #x und y Werte der abgelesenen horizontalen und vertikalen Minima
    x_h, y_h = np.genfromtxt('data_01_'+ i + '.txt', unpack = True)* 1000
    x_v, y_v = np.genfromtxt('data_01_b_'+ i + '.txt', unpack = True)* 1000
    plt.plot(x_h, y_h, 'r.', label = 'Messwerte')
    plt.plot(x_v, y_v, 'b.', label = 'Messwerte')
    params_h, cov_h = curve_fit(linear, x_h, y_h)
    params_v, cov_v = curve_fit(linear, x_v, y_v)
    plot_x_h = np.linspace(x_h[0], x_h[-1], 100)
    plt.plot(plot_x_h, linear(plot_x_h, *params_h), 'r-', label = r'Fit $\vec{a}$')
    plot_x_v = np.linspace(x_v[0], x_v[-1], 100)
    plt.plot(plot_x_v, linear(plot_x_v, *params_v), 'b-', label = r'Fit $\vec{b}$')
    plt.xlim(0, 2500)
    plt.ylim(0, 2500)
    plt.legend()
    plt.imshow(img)
    plt.xlabel(r'$x$/pm')
    plt.ylabel(r'$y$/pm')
    #plt.show()


    #Berechnungen zu den Gittervektoren
    #Arrays mit Fitparametern und Fehlern
    params_hcor = correlated_values(params_h, cov_h)
    params_vcor = correlated_values(params_v, cov_v)
    #print(params_hcor, params_vcor)
    n_h = (len(x_h) - 1)
    n_v = (len(x_v) - 1)
    #x, y Werte der ersten und letzten punkte

    delta_x_h = x_h[0] - x_h[-1]
    delta_y_h = linear(x_h[0], *params_h) - linear(x_h[-1], *params_h)
    mid_x_h = delta_x_h/n_h
    mid_y_h = delta_y_h/n_h
    #print(sqrt(mid_x_h**2 +  mid_y_h**2))
    vec_h = np.array([delta_x_h, delta_y_h])
    print(vec_h/1000, len(x_h) - 1)
    #print(norm(vec_h)/(len(x_h) - 1))

    delta_x_v = x_v[-1] - x_v[0]
    delta_y_v = linear(x_v[-1], *params_v) - linear(x_v[0], *params_v)
    mid_x_v = delta_x_v/n_v
    mid_y_v = delta_y_v/n_v
    vec_v = np.array([delta_x_v, delta_y_v])
    #print(sqrt(mid_x_v**2 +  mid_y_v**2))
    print(vec_v/1000, len(x_v) - 1)
    #print(norm(vec_v)/(len(x_v) - 1))

    theta = angle(vec_h, vec_v)
    print(theta)
    #Mittlere Vektoren

    globvec.append(vec_h/1000/n_h)
    globvec.append(vec_v/1000/n_v)



    #durchschnittliche x und y Abstände
    #d_x_h = np.array([x_h[i]-x_h[i-1] for i in range(1, len(x_h))])
    #d_y_h = np.array([y_h[i]-y_h[i-1] for i in range(1, len(y_h))])
    #d_x_v = np.array([x_v[i]-x_v[i-1] for i in range(1, len(x_v))])
    #d_y_v = np.array([y_v[i]-y_v[i-1] for i in range(1, len(y_v))])

    #Bestimmung der STreckungsfaktoren, Literaturwert a
    a = 0.246 * 1000
    s_y_2 = (   ((len(x_v)-1)*a)**2 - (delta_x_v**2 * (len(x_h)-1)*a)/((delta_x_h)**2)   )/(delta_y_v**2 - delta_y_h**2 * delta_x_v**2/delta_x_h**2)
    s_y = sqrt(s_y_2)
    print(s_y_2)
    s_x_2 = (((len(x_h)-1)*a)**2 - s_y_2 * delta_y_h**2 )/(delta_x_h**2)
    s_x = sqrt(s_x_2)
    #print(sqrt((s_x*delta_x_v)**2 + (s_y*delta_y_v)**2)/a)


    #Neuer Winkel
    vec_h_scaled = np.array([s_x * delta_x_h, s_y * delta_y_h])
    vec_v_scaled = np.array([s_x * delta_x_v, s_y * delta_y_v])
    theta_scaled = angle(vec_h_scaled, vec_v_scaled)
    #print(theta_scaled)

    #plt.plot([x_h[0], x_h[0] + noms(vec_h[0])], [linear(x_h, *noms(params_h))[0], linear(x_h, *noms(params_h))[0] + noms(vec_h[1])], 'g-')
    #plt.plot([x_v[0], x_v[0] + noms(vec_v[0])], [linear(x_v, *noms(params_v))[0], linear(x_v, *noms(params_h))[0] + noms(vec_v[1])], 'g-')
    plt.savefig('bilder/fit_01_'+ i + '.pdf')
#print(globvec)

vec_h_f = globvec[0]
vec_h_b = globvec[2]
vec_v_f = globvec[1]
vec_v_b = globvec[3]

mid_vec_h = np.array([mid([vec_h_f[0], vec_h_b[0]]), mid([vec_h_f[1], vec_h_b[1]])])
mid_vec_v = np.array([mid([vec_v_f[0], vec_v_b[0]]), mid([vec_v_f[1], vec_v_b[1]])])

print(mid_vec_h, norm(mid_vec_h))

print(mid_vec_v, norm(mid_vec_v))



a = 0.246
s_y_2 = (  a**2 * (mid_vec_h[0]**2 - mid_vec_v[0]**2)  ) / (mid_vec_v[1]**2 * mid_vec_h[0]**2 - mid_vec_h[1]**2 * mid_vec_v[0]**2   )
s_y = sqrt(s_y_2)
print(s_y)
s_x_2 = ( a**2 - s_y_2 * mid_vec_h[1]**2 ) / (mid_vec_h[0]**2)
s_x = sqrt(s_x_2)
S = np.matrix([[s_x, 0], [0, s_y]])
mid_vec_h_scaled = np.array([s_x * mid_vec_h[0], s_y * mid_vec_h[1]])
mid_vec_v_scaled = np.array([s_x * mid_vec_v[0], s_y * mid_vec_v[1]])
print(mid_vec_v_scaled)


print(acos(    (   mid_vec_h_scaled[0]* mid_vec_v_scaled[0] + mid_vec_h_scaled[1] * mid_vec_v_scaled[1]   )  /  (   sqrt(mid_vec_h_scaled[0]**2 + mid_vec_h_scaled[1]**2) * sqrt(mid_vec_v_scaled[0]**2 + mid_vec_v_scaled[1]**2)   ))/np.pi*180)

def const(x, h):
    return [h for i in x]

#AUSWERTUNG DER GOLDDATEN
plt.clf()
img = mpimg.imread('bilder/au.png')
plt.imshow(img)
plt.savefig('bilder/goldoberfläche.pdf')
x, y = np.genfromtxt('data_aut.txt', unpack = True)*1e9
x_p_1 = x[x < 15]
y_p_1 = y[x < 15]

x_p_2 = x[x > 18]
y_p_2 = y[x > 18]
plt.clf()
plt.plot(x, y)

params_p_1, cov_p_1 = curve_fit(const, x_p_1, y_p_1)
params_p_2, cov_p_2 = curve_fit(const, x_p_2, y_p_2)
h_1 = ufloat(params_p_1[0], np.sqrt(cov_p_1[0][0]))
h_2 = ufloat(params_p_2[0], np.sqrt(cov_p_2[0][0]))
print(abs(h_1 - h_2))
print(h_1)
print(h_2)
plt.clf()

x_plot_p_1 = np.linspace(0, x_p_1[-1])
x_plot_p_2 = np.linspace(x_p_2[0], x_p_2[-1])
plt.plot(x_plot_p_1, const(x_plot_p_1, *params_p_1), 'r-', label = '$h_1$')
plt.plot(x_plot_p_2, const(x_plot_p_2, *params_p_2), 'b-', label = '$h_2$')
plt.axvline(x = 15, linestyle = '--', label = 'Schranken für Plateaus')
plt.axvline(x = 18, linestyle = '--')
plt.xlim(0, x[-1])
plt.plot(x_p_1, y_p_1, 'r-')
plt.plot(x_p_2, y_p_2, 'b-')
plt.xlabel('$x$/nm')
plt.ylabel('$z$/nm')
plt.plot(x, y, 'k-', label = 'Höhenprofil')
plt.grid()
plt.legend(loc = 'best')
plt.savefig('bilder/au_plateau.pdf')
