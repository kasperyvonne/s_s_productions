import numpy as np
import matplotlib.pyplot as plt

if plt.rcParams["text.usetex"] is False:
    plt.rcParams["text.usetex"] = True

if plt.rcParams["text.latex.unicode"] is False:
    plt.rcParams["text.latex.unicode"] = True

if "siunitx" not in plt.rcParams["text.latex.preamble"]:
    plt.rcParams["text.latex.preamble"].append(r"\usepackage{siunitx}")

def g_1_g_2(L, r_1, r_2):
    return (1 - L/r_1) * (1 - L/r_2)


L = np.linspace(0, 3)

r_2 = 1.4
plt.barh(0.5 , 3, 1, 0, color="green", alpha=0.2, edgecolor="gray", label="Stabilitätsbereich")
r_1 = 1.0
plt.plot(L, g_1_g_2(L, r_1, r_2), 'b-', label = '$r_{1} = \SI{1000}{\milli\meter}$', linewidth = 1)
r_1 = 1.4
plt.plot(L, g_1_g_2(L, r_1, r_2), 'r-', label = '$r_{1} = \SI{1400}{\milli\meter}$', linewidth = 1)
plt.xlim(0, 3)
plt.legend()
plt.xlabel('$L/\si{\meter}$')
plt.ylabel('$g_1 g_2$')
plt.grid()
plt.savefig('g_1_g_2.pdf')
