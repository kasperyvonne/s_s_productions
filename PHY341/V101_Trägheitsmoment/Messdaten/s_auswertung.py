import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from collections import OrderedDict


Abstand, Zeiten = np.genfromtxt('eigenmoment.txt', unpack = True)
print(Abstand[::3])
for i in range(len(Zeiten))[::3]:
    Zeitenmittel = np.mean(Zeiten[i:i+3])
    print(Zeitenmittel)
print(Zeitenmittel)
