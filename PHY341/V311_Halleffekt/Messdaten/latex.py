class Latexdocument(object):
    def __init__(self, filename):
        self.name = filename
    def tabular(self, spalten, header, places, caption, label):
        with open(self.name, 'w') as f:
            f.write('\\begin{table} \n \\centering \n \\begin{tabular}{')
            f.write(len(spalten) * 'S ')
            f.write('} \n \\toprule  \n')
            f.write(header + '  \\\ \n')
            f.write('\\midrule  \n ')
            for i in range(0, len(spalten[0])):
                for j in range(0, len(spalten)):
                    if j == len(spalten) - 1:
                        f.write(('{:.' + str(places[j]) + 'f}' + '\\\ \n').format(spalten[j][i]))
                    else:
                        f.write(('{:.' + str(places[j]) + 'f} ' + ' & ').format(spalten[j][i]))
            f.write('\\bottomrule \n \\end{tabular} \n \\caption{' + caption + '} \n \\label{' + label + '} \n  \\end{table}')
