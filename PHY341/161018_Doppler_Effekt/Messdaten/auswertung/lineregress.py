ximport numpy as np

u,nu=np.genfromtxt('schwebung_werte_sammlung.txt',unpack=True)

u*=2
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

np.savetxt('Lingress_Mittelwert_schwebung_sammlung_2v.txt',np.column_stack([m,m_e,b,b_e]),header='m m_err b b_e')

