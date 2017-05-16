import numpy as np
import sys
from collections import OrderedDict

print('Dateinamen eingeben (ohne .txt): ')
dateiname = input()

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

np.savetxt(dateiname + '.txt', np.column_stack(data.values()), header = head)
