from load_data import *
from auswertung import *


#l.Latexdocument('tabs/methode_1.tex').tabular([noms(method_1_g), stds(method_1_g), noms(method_1_b), stds(method_1_b),
#noms(method_1_B), stds(method_1_B), noms(V_2), stds(V_2),
#noms(V_1), stds(V_1), noms(V_2)/noms(V_1) - 1],
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

#l.Latexdocument('tabs/colors.tex').tabular([np.ones(len(np.hstack([b_2_blau, b_2_rot]))), noms(np.hstack([g_1_blau, g_1_rot])), stds(np.hstack([g_1_blau, g_1_rot])), noms(np.hstack([b_1_blau, b_1_rot])), stds(np.hstack([b_1_blau, b_1_rot])),
#noms(np.hstack([g_2_blau, g_2_rot])), stds(np.hstack([g_2_blau, g_2_rot])), noms(np.hstack([b_2_blau, b_2_rot])), stds(np.hstack([b_2_blau, b_2_rot]))],
#header = '{Farbe} & \multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$} &\multicolumn{2}{c}{$g_1 \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b_1 \:/\: \si{\centi\meter}$}',
#places = [0, 1, 1, 1, 1, 1, 1, 1, 1],
#label='tab: bessel', caption='Messdaten zur Bestimmung der Brennweite mit der Methode nach Bessel.')

#V_abbe = abbe_B / G
#l.Latexdocument('tabs/abbe.tex').tabular([noms(abbe_g), stds(abbe_g), noms(abbe_b), stds(abbe_b),
#noms(abbe_B), stds(abbe_B), noms(1 + 1/V_abbe), stds(1 + 1/V_abbe), noms(1 + V_abbe), stds(1 + V_abbe)],
#header = ' \multicolumn{2}{c}{$g\' \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$b\' \:/\: \si{\centi\meter}$} &\multicolumn{2}{c}{$B \:/\: \si{\centi\meter}$} & \multicolumn{2}{c}{$1 + \\frac{1}{V}$} & \multicolumn{2}{c}{$1 + V$}',
#places = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
#label='tab: bessel', caption='Messdaten zur Bestimmung der Brennweite und Lage der Hauptebenen einer Linsenanordnung mit der Methode nach Abbe.')
