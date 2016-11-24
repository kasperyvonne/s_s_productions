import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from pint import UnitRegistry
import matplotlib.pyplot as plt
u = UnitRegistry()
Q_ = u.Quantity

#umrechnung einheiten mit var.to('unit')
# Einheiten für pint:dimensionless, meter, second, degC, kelvin, ampere, kilogram, gram, pascal, bar, speed_of_light, mol
#beispiel:
a = ufloat(5, 2) * u.meter
b = Q_(unp.uarray([5,4,3], [0.1, 0.2, 0.3]), 'cal')
c = Q_(1, 'cal')
s = Q_(1, 'second')
print(c.to('joule'))
print(a**2)
print(b)
###############################
