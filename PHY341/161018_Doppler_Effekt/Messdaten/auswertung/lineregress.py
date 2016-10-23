import numpy as np

g,nu,f=np.genfromtxt('Mittelwert_schwebung_neg.txt',unpack=True)
g_2,gm,g_mf,u,u_f=np.genfromtxt('Mittelwert_u.txt',unpack=True)
nu*=-1
u*=-1
print(nu)
print(u)
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


m,m_e,b,b_e=linregress(u,nu)

np.savetxt('Lingress_Mittelwert_schwebung_neg.txt',np.column_stack([m,m_e,b,b_e]),header='m m_err b b_e')

