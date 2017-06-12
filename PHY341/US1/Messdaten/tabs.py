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

#CEPSTRUM
l.Latexdocument('tabs/cepstrum.tex').tabular([[1, 2, 3], t_p],
header = r'{Puls} & {$t / \si{\micro\second}$}',
places = [0, 1], caption = r'Daten zur Vermessung der Acrylplatten. Zeitliche Abstände $t$ der am Cepstrum abgelesenen Impulse zum Ursprung.',
label = 'cepstrum')


#AUGE
l.Latexdocument('tabs/auge.tex').tabular([[1, 2, 3, 4, 5], t_a],
header = r'{Puls} & {$t / \si{\micro\second}$}',
places = [0, 1], caption = r'Daten zur Vermessung des Augenmodells. Zeitliche Abstände $t$ der Impulse zum Ursprung.',
label = 'auge')


#DÄMPFUNG
l.Latexdocument('tabs/dämpfung.tex').tabular([np.sort(L), U_1[np.argsort(L)], U_2[np.argsort(L)], U_2[np.argsort(L)]/U_1[np.argsort(L)]],
header = r'{$x/\si{\milli\meter}$} & {$U_1 / \si{\volt}$} & {$U_2 / \si{\volt}$} & {$\frac{U_2}{U_1}$}',
places = [1, 3, 3, 3], caption = r'Daten zur Bestimmung der Dämpfungskonstanten $\alpha$. Länge der Acrylzylinder $x$ und Spannungsamplituden des ersten bzw. zweiten Pulses $U_1$ und $U_2$.',
label = 'dämpfung')
