import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pint import UnitRegistry
import latex

u = UnitRegistry()
Q_ = u.Quantity

R = Q_(1, 'ohm')
L = Q_(1, 'henry')
C = Q_(1, 'farad')

R_ap_theo = np.sqrt(4*L/C).to('ohm')
print(R_ap_theo)



#Plotbereich

#plt.xlim()
#plt.ylim()
#aufvariabele=np.linsspace()
#
#plt.plot(,,'rx',label='')
#
#plt.grid()
#plt.legend(loc='best')
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('.pdf')
