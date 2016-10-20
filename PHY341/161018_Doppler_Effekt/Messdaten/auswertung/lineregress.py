import numpy as np


g,m,f,u,u_f=np.genfromtxt('Mittelwert_u.txt', unpack=True)

g_n,s =np.genfromtxt('Dopplereffekt_pos.txt', unpack=True)

g_p,s =np.genfromtxt('Dopplereffekt_neg.txt', unpack=True)
# Lineare Regression mittels Methode der kleinsten Quadrate
def linregress(x, y):
    assert len(x) == len(y)

    x, y = np.array(x), np.array(y)

    N = len(y)
    Delta = N * np.sum(x**2) - (np.sum(x))**2

    # A ist die Steigung, B der y-Achsenabschnitt
    A = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / Delta
    B = (np.sum(x**2) * np.sum(y) - np.sum(x) * np.sum(x * y)) / Delta

    sigma_y = np.sqrt(np.sum((y - A * x - B)**2) / (N - 2))

    A_error = sigma_y * np.sqrt(N / Delta)
    B_error = sigma_y * np.sqrt(np.sum(x**2) / Delta)

    return A, A_error, B, B_error


m,m_e,b,b_e=linregress(u,g_p)

np.savetxt('Lingress_berechneter_doppler_pos.txt',np.column_stack([m,m_e,b,b_e]),header='m m_err b b_e')

