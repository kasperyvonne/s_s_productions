from auswertung import *

#l.Latexdocument('tabs/daten_V503.tex').tabular([resistance.magnitude, T.magnitude, eta.magnitude*1e5, r_korr.to('micrometer').magnitude, time.magnitude, v.magnitude,
# voltage.magnitude, q.magnitude*10**19  ],
#'{$R/\si{\mega\ohm}$} & {$T/\si{\celsius}$} & {$\eta/10^{-5}\si{\\newton\second\meter^{-2}}$} & {$r/\si{\micro\meter}$} & {$t/\\si{\second}$} &\n {$v_0/\\si{\centi\meter\second^{-1}}$} & {$U/\si{\\volt}$}  & {$q/10^{-19}\si{\coulomb}$}',
#   [2, 2, 2, 1, 3, 3, 2, 2, 2], label = 'data', caption = 'Gemessene und brechnete Größen für einzelne beobachtete Tropfen. Thermowiderstand $R$, Temperatur $T$, Luftviskosität $\eta$, Tröpfchenradius $r$ (korrigiert), Fallzeit $t$, Fallgeschwindigkeit $v_0$, Schwebespannung $U$ und Ladung $q$.')

#l.Latexdocument('tabs/thermowiderstand.tex').tabular([R_fit[:len(R_fit)/2], T_fit[:len(R_fit)/2], R_fit[len(R_fit)/2 : ], T_fit[len(R_fit)/2 : ] ],
#header = '{$R /\si{\mega\ohm}$} & {$T/\si{\celsius}$} & {$R /\si{\mega\ohm}$} & {$T/\si{\celsius}$}',
#places = [2, 0, 2, 0],
#label = 'thermowiderstand', caption = 'Wertepaare zur Interpolation des Zusammenhangs zwischen $R$ und $T$.')

#l.Latexdocument('tabs/q_best.tex').tabular([q_sort[0:10]*1e19, q_test_best[0:10]*1e19, q_sort[10:]*1e19, q_test_best[10:]*1e19],
#header = '{$q_{\mathup{i}} /\SI{e-19}{\coulomb}$} & {$\mathup{e}_{\mathup{i}} /\SI{e-19}{\coulomb}$} & {$q_{\mathup{i}} /\SI{e-19}{\coulomb}$} & {$\mathup{e}_{\mathup{i}} /\SI{e-19}{\coulomb}$}',
#places = [2, 2, 2, 2],
#label = 'q_best', caption = 'Verwendete Tröpfchenladungen $q_{\mathup{i}}$ zur Bestimmung der Elementarladung und jeweils berechnete Minimalstelle $\mathup{e}_{\mathup{i}}$ der Gleichung \eqref{eq: rundung}.')
