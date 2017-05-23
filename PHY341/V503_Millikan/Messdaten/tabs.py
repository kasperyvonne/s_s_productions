from auswertung import *

l.Latexdocument('tabs/daten_V503.tex').tabular([resistance.magnitude, T.magnitude, eta.magnitude, radius.magnitude, time.magnitude, v.magnitude,
 voltage.magnitude, q.magnitude*10**11],
'{$R/\si{\mega\ohm}$} & {$T/\si{\celsius}$} & {$\eta/10^{-5}\si{\\newton\second\meter^{-2}}$} & {$r/\si{\milli\meter}$} & {$t/\\si{\second}$} & {$v_0/\\si{\centi\meter\second^{-1}}$} & {$U/\si{\\volt}$}  & {$q/10^{-11}\si{\coulomb}$}',
   [2, 2, 2, 2, 3, 3, 2, 2, 2], 'test', 'test')
