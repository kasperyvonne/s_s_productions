import numpy as np
import sys



a = []
b = []
c = []
d = []
e = []


u = 'a'

while u != 'x':
    u = input()
    if u == 'x':
        break

    v = float(u)
    a.append(v)
    print('a:  ', a)
    w = float(input())
    b.append(w)
    print('b:  ', b)

    x = float(input())
    c.append(x)
    print('c:  ', c)

    y = float(input())
    d.append(y)
    print('d:  ', d)

    z = float(input())
    e.append(z)
    print('e:  ', e)




np.savetxt('daten.txt', np.column_stack([a, b, c, d, e]), header="a b c d e")
