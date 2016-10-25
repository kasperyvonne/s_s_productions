import numpy as np


# Zunächst für die Länge l = 0.7 m

t_links = np.genfromtxt('T_Pendel_links_70.txt', unpack = True)
t_links_mid = np.mean(t_links)
t_links_std = 1/np.sqrt(len(t_links)) * np.std(t_links)
print(t_links_mid, t_links_std)


t_rechts = np.genfromtxt('T_Pendel_rechts_70.txt', unpack = True)
t_rechts_mid = np.mean(t_rechts)
t_rechts_std = 1/np.sqrt(len(t_rechts)) * np.std(t_rechts)
print(t_rechts_mid, t_rechts_std)
