import sympy

def error(f, err_vars=None):
    from sympy import Symbol, latex
    s = 0
    latex_names = dict()

    if err_vars == None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'

    return latex(sympy.sqrt(s), symbol_names=latex_names)


## Fehlerfortpflanzung Addition
t_1, t_2 = sympy.var('t_1 t_2')
f = t_1+t_2


print('-------------------- \n Fehlerformel von: \n ',f, '\n')
print(error(f))
print('--------------------\n')


##Fehlerfortpflazung lograitmierter Druck

p, p_0, p_g = sympy.var('p p_0 p_g')
g = sympy.log( (p - p_g)/ (p_0-p_g))


print('-------------------- \n Fehlerformel von: \n ',g, '\n')
print(error(g))
print('--------------------\n')

##Fehlerfortpflazung vom Zylinderfolumen

r, h = sympy.var('r h')
V = sympy.pi*r**2*h


print('-------------------- \n Fehlerformel von: \n ',V, '\n')
print(error(V))
print('--------------------\n')


##Fehlerfortpflazung vom Saugvermögen der druckkurve

m, v = sympy.var('m v')
s_p= -m*v


print('-------------------- \n Fehlerformel von: \n ',s_p, '\n')
print(error(s_p))
print('--------------------\n')

##Fehlerfortpflazung vom Saugvermögen der Leckkrate

m, p_g, V = sympy.var('m p_g V')
s_l= V *m/p_g


print('-------------------- \n Fehlerformel von: \n ',s_l, '\n')
print(error(s_l))
print('--------------------\n')


## Fehlerfortpflanzung für den Leitwert

S_0,S_eff= sympy.var('S_0 S_eff')
l= S_eff*S_0/(S_0-S_eff)


print('-------------------- \n Fehlerformel von: \n ',l, '\n')
print(error(l))
print('--------------------\n')
