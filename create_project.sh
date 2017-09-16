read -p "Enter Projektname: " dir_name
read -p "Enter Versuchstitel: " title
read -p "Enter Versuchsnummer: " vnr
read -p "Enter Tag der Durchführung: " date_durchfuehrung
read -p "Enter Tag der Abgabe: " date_abgabe

cp -avr Vorlagen/Ordnerstruktur_Praktikum PHY641
mv PHY641/Ordnerstruktur_Praktikum PHY641/$dir_name

echo "\newcommand{\versuch}{$title}" >> PHY641/$dir_name/Protokoll/data.tex
echo "\newcommand{\vnr}{$vnr}" >> PHY641/$dir_name/Protokoll/data.tex
echo "\newcommand{\vd}{Tag der Durchführung: $date_durchfuehrung}" >> PHY641/$dir_name/Protokoll/data.tex
echo "\newcommand{\va}{Tag der Abgabe: $date_abgabe}" >> PHY641/$dir_name/Protokoll/data.tex
