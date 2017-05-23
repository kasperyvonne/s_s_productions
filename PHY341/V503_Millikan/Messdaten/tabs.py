from auswertung import *
#
l.Latexdocument('tabs/daten_V503.tex').tabular([resistance.magnitude, T.magnitude, eta.magnitude, radius.magnitude, time.magnitude, v.magnitude,
 voltage.magnitude, q.magnitude*10**19  ],
'{$R/\si{\mega\ohm}$} & {$T/\si{\celsius}$} & {$\eta/10^{-5}\si{\\newton\second\meter^{-2}}$} & {$r/\si{\milli\meter}$} & {$t/\\si{\second}$} &\n {$v_0/\\si{\centi\meter\second^{-1}}$} & {$U/\si{\\volt}$}  & {$q/10^{-19}\si{\coulomb}$}',
   [2, 2, 2, 2, 3, 3, 2, 2, 2], label = 'data', caption = 'Gemessene und brechnete Größen für einzelne beobachtete Tropfen. Thermowiderstand $R$, Temperatur $T$, Luftviskosität $\eta$, Tröpfchenradius $r$, Fallzeit $t$, Fallgeschwindigkeit $v_0$, Schwebespannung $U$ und korrigierte Ladung $q$.')

#l.Latexdocument('tabs/thermowiderstand.tex').tabular([R_fit[:len(R_fit)/2], T_fit[:len(R_fit)/2], R_fit[len(R_fit)/2 : ], T_fit[len(R_fit)/2 : ] ],
#header = '{$R /\si{\mega\ohm}$} & {$T/\si{\celsius}$} & {$R /\si{\mega\ohm}$} & {$T/\si{\celsius}$}',
#places = [2, 0, 2, 0],
#label = 'thermowiderstand', caption = 'Wertepaare zur Interpolation des Zusammenhangs zwischen $R$ und $T$.')
