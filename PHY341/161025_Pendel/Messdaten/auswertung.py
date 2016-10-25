import numpy as np


# Zunächst für die Länge l = 0.7 m

t_links = 0.2 * np.genfromtxt('T_Pendel_links_70.txt', unpack = True)
t_links_mid = np.mean(t_links)
t_links_std = np.
