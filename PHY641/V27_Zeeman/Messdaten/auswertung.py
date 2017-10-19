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
import matplotlib.image as mpimg
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

#FITFUNKTIONEN
def hysterese(I, a, b, c, d):
    return a * I**3 + b * I**2 + c * I + d




#######LOAD DATA#########
B_up, B_down = np.genfromtxt('data/hysterese.txt', unpack=True) #Flussdichte bei auf und absteigendem Spulenstrom
I = np.linspace(0, 20, 21) # Stromstärke von 0 bis 20A in 1A Schritten


###FIT DER HYSTERESE######
params_up, cov_up = curve_fit(hysterese, I, B_up)
params_down, cov_down = curve_fit(hysterese, I, B_down)






#######PLOTS#########
plt.plot(I, B_up, 'rx', label = 'Messwerte')
I_plot = np.linspace(-1, 21, 100)
plt.plot(I_plot, hysterese(I_plot, *params_up))
plt.xlim(-1, 21)
plt.xlabel('$I/$A')
plt.ylabel('$B/$mT')
plt.legend()
plt.grid()
plt.savefig('plots/hysterese_aufsteigend.pdf')

plt.clf()
plt.plot(I, B_down, 'rx', label = 'Messwerte')
plt.plot(I_plot, hysterese(I_plot, *params_down))
plt.xlim(-1, 21)
plt.xlabel('$I/$A')
plt.ylabel('$B/$mT')
plt.legend()
plt.grid()
plt.savefig('plots/hysterese_absteigend.pdf')


#r.makeresults()


plt.clf()

#print(img)
plt.gray()
img= mpimg.imread('bilder_v27/messung_3_blau_pi/IMG_0648.JPG')
lum_img = img[:,:,1]
plt.hist(lum_img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

#plt.imshow(img)
plt.show()
