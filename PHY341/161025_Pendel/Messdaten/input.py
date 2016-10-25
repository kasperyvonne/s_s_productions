import numpy as np
import sys
from collections import OrderedDict

print('Anzahl der Messgrößen eingeben')
p = int(input())
data = OrderedDict()


head = ""
for i in range(0, p):
    print(i+1, 'te Messgröße eingeben')
    v = input()
    data[v] = []
    head += v
    head += " "

print(head)
print('Zeilenweise die Messwerte eingeben: ')
u = 'a'
while u != 'x':
    for k in data:
        u = input()
        if u == 'x':
            break
        data[k].append(float(u))
        print(data[k])

np.savetxt('T_schwebe_70.txt', np.column_stack(data.values()), header = head)





with open('T_schwebe_70.tex', 'w') as f:


    f.write('\\begin{table} \n \\centering \n \\caption{...} \n \\label{tab:...} \n \\begin{tabular}{')
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
