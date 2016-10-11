import numpy as np


a, b, c, d, e = np.genfromtxt('LED.txt', unpack=True)

leng = len(a)
i = 0




with open('tabelle.tex', 'w') as f:


    f.write(r'\input{header.tex}')
    f.write('\\begin{document} \n')

    f.write('\\begin{table} \n \\centering \n \\caption{Testtabelle} \n \\label{tab:some_data} \n \\begin{tabular}{S S S S S} \n \\toprule \\\ \n $\\alpha$ & $\\beta$ & $\\gamma$ & $\\theta$ & $\\kappa$ \\\ \n  \\midrule \\\ \n ')

    while i< leng:
        f.write('{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\ \n '.format(a[i], b[i], c[i], d[i], e[i]))
        i += 1



    f.write('\\bottomrule \n \\end{tabular} \n \\end{table}')

    f.write('\\end{document} \n ')
