from pandas import Series, DataFrame
import pandas as pd
import collections
import numpy
import uncertainties
import pint
from uncertainties import ufloat
from uncertainties import ufloat_fromstr
from pint import UnitRegistry
import string
ureg = UnitRegistry()
Q_ = ureg.Quantity


class Latexdocument(object):
    def __init__(self, filename):
        self.name = filename
        self.data = DataFrame(columns=(['tex', 'var']))
    def tabular(self, spalten, header, places, caption, label):
        with open(self.name, 'w') as f:
            f.write('\\begin{table} \n\\centering \n\\caption{' + caption + '} \n\\label{tab: ' + label + '} \n\\begin{tabular}{')
            f.write(len(spalten) * 'S ')
            f.write('} \n\\toprule  \n')
            f.write(header + '  \\\ \n')
            f.write('\\midrule  \n ')
            for i in range(0, len(spalten[0])):
                for j in range(0, len(spalten)):
                    if j == len(spalten) - 1:
                        f.write(('{:.' + str(places[j]) + 'f}' + '\\\ \n').format(spalten[j][i]))
                    else:
                        f.write(('{:.' + str(places[j]) + 'f} ' + ' & ').format(spalten[j][i]))
            f.write('\\bottomrule \n\\end{tabular} \n\\end{table}')

    def app(self, name, value):
            if (type(value.magnitude) == uncertainties.core.Variable or type(value.magnitude) == uncertainties.core.AffineScalarFunc):
                val = '{:+.1uS}'.format(value.magnitude)
                s = '{:Lx}'.format(Q_(2, value.units)) + '~'
                df = DataFrame(collections.OrderedDict({'var': pd.Series(value, index = [name] ),
                #'tex': name + ' = \SI{' + val[:val.index('+')]+ ' \pm ' + val[val.index('-')+1:] + s[s.index('}{'):s.index('~')]}))
                'tex': name + ' = \SI{' + val + '}{' + s[s.index('}{'):s.index('~')]}))
                self.data = self.data.append(df)
            else:
                df = DataFrame({'var': pd.Series(value, index = [name] ),
                'tex': name + ' = ' + '{:Lx}'.format(value)})
                self.data = self.data.append(df)


    def makeresults(self):
        print(self.data['var'])
        with open(self.name, 'w') as f:
            for i in self.data['tex']:
                f.write(i + '\n')
