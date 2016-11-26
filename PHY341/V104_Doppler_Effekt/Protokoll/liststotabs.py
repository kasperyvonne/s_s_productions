import numpy as np
import sys
from collections import OrderedDict

print('Anzahl der Messgrößen eingeben')
p = int(input())
data = OrderedDict()


for i in range(0, p):
    print(i+1, 'te Messgröße eingeben')
    v = input()
    data[v] = []

# Vor Programmstart muss Dateipfad und Name der .tex Datei angepasst werden
array = np.array(np.genfromtxt(r'/home/stefan/Desktop/s_s_productions/PHY341/161018_Doppler_Effekt/Messdaten/frequenz_v_neg.txt', unpack = True))
helpindex = 0
for k in data:
    data[k] = array[helpindex]
    helpindex+=1


with open('tab.tex', 'w') as f:


    f.write('\\begin{table} \n \\centering \n \\caption{Testtabelle} \n \\label{tab:some_data} \n \\begin{tabular}{')
    f.write(p*'S ')
    f.write('} \n \\toprule \\\ \n')

    helpindex = 0
    for key in data.keys():
        if helpindex == p-1:
            f.write('$'+ key + '$ \\\ \n')
        else:
            f.write('$' + key + '$  & ')
        helpindex +=1




    f.write('\\midrule \\\ \n ')

    rowCount = len(data[v])
    for j in range(0, rowCount):
        helpindex = 0
        for i in data:

            if helpindex == p-1:
                f.write('{:.2f} \\\ \n '.format(data[i][j]))
            else:
                f.write('{:.2f} & '.format(data[i][j]))
            helpindex += 1


    f.write('\\bottomrule \n \\end{tabular} \n \\end{table}')
