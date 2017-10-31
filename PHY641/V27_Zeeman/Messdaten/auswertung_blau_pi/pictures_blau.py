import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
from scipy import signal
import matplotlib.image as mpimg

blau_pi_I_0 = scipy.misc.imread('../bilder_v27/messung_3_blau_pi/bw/m3_I_0_cut-LAB.png', flatten=True, mode=None)
blau_pi_I_17 = scipy.misc.imread('../bilder_v27/messung_3_blau_pi/bw/m3_I_17_cut-LAB.png', flatten=True, mode=None)

plt.plot(range(0, 4000), blau_pi_I_0[450], 'r-', linewidth = 0.5, label = '$I = 0$')
plt.plot(range(0, 4000), blau_pi_I_17[450], 'b-', linewidth = 0.5, label = '$I = 17$A')
plt.xlim(0, 4000)
plt.ylim(60, 160)
plt.xlabel('$x$/px')
plt.ylabel('Helligkeit')
plt.legend()
plt.grid()
plt.savefig('../plots/blau_pi_intensitaet.pdf')



plt.clf()

plt.subplot(211)
plt.plot(range(0, 4000), blau_pi_I_0[450], 'r-', linewidth = 0.5)
peaks_blau_0 = np.genfromtxt('../data/peaks_blau_pi_I_0.txt', unpack = True)
for i in peaks_blau_0:
    plt.axvline(x = i, linewidth=0.7, color='k')
plt.ylabel('Helligkeit')
plt.xlabel('$x$/px')
plt.xlim(0, 4000)

plt.subplot(212)
img = mpimg.imread('../bilder_v27/messung_3_blau_pi/m3_I_0_cut.png')
plt.imshow(img)
for i in peaks_blau_0:
    plt.axvline(x = i, linewidth=0.7, color='w')
plt.xlabel('$x$/px')
plt.savefig('../plots/peaks_blau_pi_0.pdf')

plt.clf()
plt.subplot(211)
plt.plot(range(0, 4000), blau_pi_I_17[450], 'b-', linewidth = 0.5)
peaks_blau_17 = np.genfromtxt('../data/peaks_blau_pi_I_17.txt', unpack = True)
for i in peaks_blau_17:
    plt.axvline(x = i, linewidth=0.7, color='k')
plt.xlim(0, 4000)
plt.ylabel('Helligkeit')
plt.xlabel('$x$/px')

plt.subplot(212)
img = mpimg.imread('../bilder_v27/messung_3_blau_pi/m3_I_17_cut.png')
plt.imshow(img)
for i in peaks_blau_17:
    plt.axvline(x = i, linewidth=0.7, color='w')
plt.xlabel('$x$/px')
plt.savefig('../plots/peaks_blau_pi_17.pdf')
