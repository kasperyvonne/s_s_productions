import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

# Zunächst für die Länge l = 0.7 m

t_links = np.genfromtxt('T_Pendel_links_70.txt', unpack = True)
t_links_mid_70 = np.mean(t_links)
t_links_std_70 = 1/np.sqrt(len(t_links)) * np.std(t_links)
u_t_links_70 = ufloat(t_links_mid_70, t_links_std_70)
print(u_t_links_70)

t_rechts = np.genfromtxt('T_Pendel_rechts_70.txt', unpack = True)
t_rechts_mid_70 = np.mean(t_rechts)
t_rechts_std_70 = 1/np.sqrt(len(t_rechts)) * np.std(t_rechts)
u_t_rechts_70 = ufloat(t_rechts_mid_70, t_rechts_std_70)
print(u_t_rechts_70)


t_gegen = np.genfromtxt('T_gegensinnig_70.txt', unpack = True)
t_gegen_mid_70 = np.mean(t_gegen)
t_gegen_std_70 = 1/np.sqrt(len(t_gegen)) * np.std(t_gegen)
u_t_gegen_70 = ufloat(t_gegen_mid_70, t_gegen_std_70)
print(u_t_gegen_70)


t_gleich = np.genfromtxt('T_gleichsinnig_70.txt', unpack = True)
t_gleich_mid_70 = np.mean(t_gleich)
t_gleich_std_70 = 1/np.sqrt(len(t_gleich)) * np.std(t_gleich)
u_t_gleich_70 = ufloat(t_gleich_mid_70, t_gleich_std_70)
print(u_t_gleich_70)


t_gekop = np.genfromtxt('T_gekoppelt_70.txt', unpack = True)
t_gekop_mid_70 = np.mean(t_gekop)
t_gekop_std_70 = 1/np.sqrt(len(t_gekop)) * np.std(t_gekop)
u_t_gekop_70 = ufloat(t_gekop_mid_70, t_gekop_std_70)
print(u_t_gekop_70)

t_schwebe = np.genfromtxt('T_schwebe_70.txt', unpack = True)
t_schwebe_mid_70 = np.mean(t_schwebe)
t_schwebe_std_70 = 1/np.sqrt(len(t_schwebe)) * np.std(t_schwebe)
u_t_schwebe_70 = ufloat(t_schwebe_mid_70, t_schwebe_std_70)
print(u_t_schwebe_70)

Werte = {"T_{S}": (u_t_schwebe_70.n, u_t_schwebe_70.s), "T_{S}": (u_t_schwebe_70.n, u_t_schwebe_70.s), "T_{1}": (u_t_links_70.n, u_t_links_70.s), "T_{+}": (u_t_gleich_70.n, u_t_gleich_70.s),
"T_{-}": (u_t_gegen_70.n, u_t_gegen_70.s) ,"T": (u_t_gekop_70.n, u_t_gekop_70.s), "T_{2}": (u_t_rechts_70.n, u_t_rechts_70.s)}

for key in Werte:
    print(Werte[key])


# Nun für Länge l = 0.6 m

t_links =  np.genfromtxt('T_Pendel_links_60.txt', unpack = True)
t_links_mid_60 = np.mean(t_links)
t_links_std_60 = 1/np.sqrt(len(t_links)) * np.std(t_links)
u_t_links_60 = ufloat(t_links_mid_60, t_links_std_60)
print(u_t_links_60)


t_rechts = np.genfromtxt('T_Pendel_rechts_60.txt', unpack = True)
t_rechts_mid_60 = np.mean(t_rechts)
t_rechts_std_60 = 1/np.sqrt(len(t_rechts)) * np.std(t_rechts)
u_t_rechts_60 = ufloat(t_rechts_mid_60, t_rechts_std_60)
print(u_t_rechts_60)


t_gegen = np.genfromtxt('T_gegensinnig_60.txt', unpack = True)
t_gegen_mid_60 = np.mean(t_gegen)
t_gegen_std_60 = 1/np.sqrt(len(t_gegen)) * np.std(t_gegen)
u_t_gegen_60 = ufloat(t_gegen_mid_60, t_gegen_std_60)
print(u_t_gegen_60)


t_gleich = np.genfromtxt('T_gleichsinnig_60.txt', unpack = True)
t_gleich_mid_60 = np.mean(t_gleich)
t_gleich_std_60 = 1/np.sqrt(len(t_gleich)) * np.std(t_gleich)
u_t_gleich_60 = ufloat(t_gleich_mid_60, t_gleich_std_60)
print(u_t_gleich_60)


t_gekop = np.genfromtxt('T_gekoppelt_60.txt', unpack = True)
t_gekop_mid_60 = np.mean(t_gekop)
t_gekop_std_60 = 1/np.sqrt(len(t_gekop)) * np.std(t_gekop)
u_t_gekop_60 = ufloat(t_gekop_mid_60, t_gekop_std_60)
print(u_t_gekop_60)


t_schwebe = np.genfromtxt('T_schwebe_60.txt', unpack = True)
t_schwebe_mid_60 = np.mean(t_schwebe)
t_schwebe_std_60 = 1/np.sqrt(len(t_schwebe)) * np.std(t_schwebe)
u_t_schwebe_60 = ufloat(t_schwebe_mid_60, t_schwebe_std_60)
print(u_t_schwebe_60)



#------------------------------------------------------------------
# berechne Wert für Schwebung mit den anderen Zeiten

t_schwebe_calc_70 =0.2 *  (u_t_links_70 * u_t_rechts_70) / (u_t_gleich_70 - u_t_gegen_70)
print('Berechneter Wert für Ts(70): ', t_schwebe_calc_70)
print('Gemessener Wert für Ts(70): ', u_t_schwebe_70)
print('Prozentuale Abweichung(70): ', t_schwebe_calc_70/u_t_schwebe_70-1)



t_schwebe_calc_60 =0.2 * (u_t_links_60 * u_t_rechts_60) / (u_t_gleich_60 - u_t_gegen_60)
print('Berechneter Wert für Ts(60): ', t_schwebe_calc_60)
print('Gemessener Wert für Ts(60): ', u_t_schwebe_60)
print('Prozentuale Abweichung(60): ', (t_schwebe_calc_60 / u_t_schwebe_60) - 1)
