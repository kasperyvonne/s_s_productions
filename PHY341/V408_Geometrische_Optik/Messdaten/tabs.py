from load_data import *
from auswertung import *


l.Latexdocument('tabs/methode_1.tex').tabular([method_1_g, method_1_b,
method_1_B, V_2, V_1,  V_2/V_1 - 1, method_1_f],
header = '{$g/\si{\centi\meter}$} & {$b/\si{\centi\meter}$} & {$B/\si{\centi\meter}$} & {$V_1$} & {$V_2$} & {$\\frac{V_2}{V_1} - 1$} & {$f/\si{\centi\meter}$}', places = [1, 1, 1, 1, 1, 2, 1],
label='methode_1', caption='Messdaten zur Überprüfung der Abbildungsgleichung \eqref{eq: abbildungsgesetz_gross} und der Linsengleichung \eqref{eq: linsengleichung}.')

l.Latexdocument('tabs/wasserlinse.tex').tabular([wasser_g, wasser_b, wasser_f],
header = '{$g / \si{\centi\meter}$} & {$b / \si{\centi\meter}$} & {$f / \si{\centi\meter}$}', places = [1, 1, 1],
label='wasserlinse', caption='Messdaten zur Bestimmung der Brennweite einer unbekannten Linse, Gegenstandsweite $g$, Bildweite $b$ und berechnte Brennweite $f$.')
#
leng = len(method_bessel_f)
l.Latexdocument('tabs/bessel.tex').tabular([bessel_g_1, bessel_b_1,
bessel_g_2, bessel_b_2, method_bessel_f[:leng/2], method_bessel_f[leng/2:]],
header = '{$g_1 / \si{\centi\meter}$} & {$b_1 / \si{\centi\meter}$} &{$g_2 / \si{\centi\meter}$} &  {$b_2 / \si{\centi\meter}$} & {$f_1 / \si{\centi\meter}$} & {$f_2 / \si{\centi\meter}$}',
places = [1, 1, 1, 1, 1, 1],
label='bessel', caption='Messdaten zur Bestimmung der Brennweite mit der Methode nach Bessel.')
#leng_2 = len(np.hstack([brennweite_blau, brennweite_rot]))
#l.Latexdocument('tabs/colors.tex').tabular([np.ones(len(np.hstack([b_2_blau, b_2_rot]))), np.hstack([g_1_blau, g_1_rot]), np.hstack([b_1_blau, b_1_rot]),
#np.hstack([g_2_blau, g_2_rot]), np.hstack([b_2_blau, b_2_rot]), np.hstack([brennweite_blau, brennweite_rot])[:leng/2], np.hstack([brennweite_blau, brennweite_rot])[leng/2:]],
#header = '{Farbe} & {$g_1 / \si{\centi\meter}$} & {$b_1 / \si{\centi\meter}$} &{$g_2 / \si{\centi\meter}$} & {$b_2 / \si{\centi\meter}$} & {$f_1 / \si{\centi\meter}$}  & {$f_2 / \si{\centi\meter}$}',
#places = [0, 1, 1, 1, 1, 1, 1],
#label='tab: bessel', caption='Messdaten zur Untersuchung der chromatischen Abberation mit der Methode nach Bessel.')
#
V_abbe = abbe_B / G
l.Latexdocument('tabs/abbe.tex').tabular([abbe_g, abbe_b, abbe_B, 1 + 1/V_abbe, 1 + V_abbe],
header = '{$g\' / \si{\centi\meter}$} & {$b\' / \si{\centi\meter}$} &{$B / \si{\centi\meter}$} & {$1 + \\frac{1}{V}$} & {$1 + V$}',
places = [1, 1, 1, 1, 1],
label='abbe', caption='Messdaten zur Bestimmung der Brennweite und Lage der Hauptebenen einer Linsenanordnung mit der Methode nach Abbe.')
