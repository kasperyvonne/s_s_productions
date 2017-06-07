from auswertung import *

#GESCHWINDIGKEITSMESSUNG MIT ECHO VERFAHREN
l.Latexdocument('tabs/c_echo.tex').tabular([np.sort(s_e), t_1_e[np.argsort(s_e)], t_2_e[np.argsort(s_e)], (delta_t/2)[np.argsort(s_e)]],
header = r'{$s / \si{\milli\meter}$} & {$t_1 / \si{\micro\second}$} &  {$t_2 / \si{\micro\second}$}& {$\frac{\Delta t }{2} / \si{\micro\second}$}',
places = [2, 1, 1, 1, 1], caption = r'Daten zur Bestimmung der Schallgeschwindigkeit in Acryl mit der Impuls-Echo-Methode. Laufstrecke $s$, Zeitlicher Abstand der Pulse zum Ursprung $t_1$, $t_2$ und berechnete halbe Zeitdifferenz $\frac{\Delta t }{2}$.',
label = 'c_echo')


#GESCHWINDIGKEITSMESSUNG MIT DURCHSCHALLUNGSVERFAHREN
l.Latexdocument('tabs/c_durchschallung.tex').tabular([np.sort(s_d), t_d[np.argsort(s_d)]],
header = r'{$s / \si{\milli\meter}$} & {$t / \si{\micro\second}$}',
places = [2, 1], caption = r'Daten zur Bestimmung der Schallgeschwindigkeit in Acryl mit der Durchschallungsmethode. Laufstrecke $s$ und Laufzeit $t$.',
label = 'c_durchsschallung')
