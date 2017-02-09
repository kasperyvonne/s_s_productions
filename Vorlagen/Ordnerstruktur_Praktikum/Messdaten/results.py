import uncertainties
from uncertainties import ufloat
class Results(object):
    def __init__(self):
        self.values = []
        self.units = []
        self.names = []

    def appendexp(self, name, value, unit):
        self.values.append(value)
        self.names.append(name)
        self.units.append(unit)

    def makeresults(self):
        def formaterror(u):
            count = 0
            n = u.n
            print(n)
            s = u.s
            if(s/10 < 1):
                if(s > 1):
                    return (str(round(n, count)) + '(' + str(int(round(s*10, 2))) + ')')
                while(s/10 < 1):
                    print(s)
                    if(s < 1):
                        print(s)
                        s *= 10
                        count += 1
                    else:
                        print(n)
                        return (str(round(n, count+1)) + '(' + str(int(round(s, 1))) + ')')
                return (str(round(n, count)) + '(' + str(int(round(s, 2))) + ')')
            elif(s/100 > 1):
                while(s/100 > 1):
                    s /= 10
                    count += 1
                if(count != 0):
                        return (str(int(round(n/(10**count), count))) + '(' + str(int(round(s, 2))) + ')' + 'e' + str(count))
                else:
                    return (str(int(round(n, count))) + '(' + str(int(round(s, 2))) + ')')
            else:
                return (str(int(round(n, count))) + '(' + str(int(round(s, 0))) + ')')
        with open('results.txt', 'w') as f:
            for i in range(0, len(self.values)):
                if(type(self.values[i]) != uncertainties.core.Variable):
                    f.write(('{} = \SI{{{}}}{{\{}}} \n').format(self.names[i], self.values[i], self.units[i]))

                if(type(self.values[i]) == uncertainties.core.Variable):
                    f.write(('{} = \SI{{{}}}{{\{}}} \n').format(self.names[i], formaterror(self.values[i]), self.units[i]))
