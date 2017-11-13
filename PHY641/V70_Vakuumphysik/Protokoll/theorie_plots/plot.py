import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-0.5, 10, 100)
p_0 = 5
p_g = 1

plt.plot(t, (p_0-p_g)*np.exp(-t) + p_g, 'r-',label = 'Druckverlauf $p(t)$')
plt.axhline(y = p_g, color = 'k', linestyle = '--')
plt.yticks([p_g, p_0], ['$p_G$', '$p_0$'])
plt.xlabel(r'$\frac{S}{V}\,t$', fontsize = 12)
plt.ylabel(r'$p$', fontsize = 12)
plt.ylim(0, p_0 + 0.5)
plt.xlim(-0.5, 7)
plt.legend()
plt.savefig('theo_p.pdf')
