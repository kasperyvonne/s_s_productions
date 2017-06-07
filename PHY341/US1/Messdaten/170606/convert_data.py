import numpy as np


with open('auge_data.dat', 'r') as r:
    with open('auge_data.txt', 'w') as w:
        for line in r:
            w.write(line.replace(',', '.').replace('E', 'e'))
