import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 100)
plt.plot(t, (2-1)*np.exp(-t) + 1)
plt.axhline(y = 1, color = 'k', linestyle = '--')
plt.yticks([1, 2], ['$p_G$', '$p_0$'])
plt.xlabel(r'$\frac{S}{V}t$')
plt.ylabel(r'$p$')
plt.savefig('theo_p.pdf')
