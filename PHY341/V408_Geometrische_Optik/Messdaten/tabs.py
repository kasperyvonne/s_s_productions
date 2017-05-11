from load_data import *
from auswertung import *


#l.Latexdocument('tabs/methode_1.tex').tabular([noms(method_1_g), stds(method_1_g), noms(method_1_b), stds(method_1_b),
#noms(method_1_B), stds(method_1_B), noms(V_1), stds(V_2),
#noms(V_2), stds(V_2), noms(V_2)/noms(V_1) - 1],
#header = '{$g/\si{\centi\meter}$} & {$b/\si{\centi\meter}$} & {$B/\si{\centi\meter}$} & {$V_1$} & {$V_2$} & {$\\frac{V_2}{V_1} - 1$}', places = [1, 1, 1, 1, 1, 1,2,2,2,2, 2],
#label='tab: methode_1', caption='Messdaten zur Überprüfung der Abbildungsgleichung \eqref{} und....')

#l.Latexdocument('tabs/wasserlinse.tex').tabular([noms(wasser_g), stds(wasser_g), noms(wasser_b), stds(wasser_b)],
#header = '\multicolumn{2}{c}{$g \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b \:/\: \si{\centi\meter}$}', places = [1, 1, 1, 1],
#label='tab: wasserlinse', caption='Messdaten zur Bestimmung der Brennweite einer unbekannten Linse, Gegenstandsweite $g$ und Bildweite $b$.')

#l.Latexdocument('tabs/bessel.tex').tabular([noms(bessel_g_1), stds(bessel_g_1), noms(bessel_b_1), stds(bessel_b_1),
#noms(bessel_g_2), stds(bessel_g_2), noms(bessel_b_2), stds(bessel_b_2)],
#header = '\multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$} &\multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$}',
#places = [1, 1, 1, 1, 1, 1, 1, 1],
#label='tab: bessel', caption='Messdaten zur Bestimmung der Brennweite mit der Methode nach Bessel.')

l.Latexdocument('tabs/colors.tex').tabular([noms(bessel_g_1), stds(bessel_g_1), noms(bessel_b_1), stds(bessel_b_1),
noms(bessel_g_2), stds(bessel_g_2), noms(bessel_b_2), stds(bessel_b_2)],
header = '\multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$} &\multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$}',
places = [1, 1, 1, 1, 1, 1, 1, 1],
label='tab: bessel', caption='Messdaten zur Bestimmung der Brennweite mit der Methode nach Bessel.')
