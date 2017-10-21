import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.misc
from scipy import signal
import matplotlib.image as mpimg

rot_I_0 = scipy.misc.imread('bilder_v27/messung_1_rot_sigma/b_w/m1_I_0_cut-LAB.png', flatten=True, mode=None)
#rot_I_7_5 = scipy.misc.imread('bilder_v27/messung_1_rot_sigma/b_w/m1_I_7_5_cut-LAB.png', flatten=True, mode=None)
rot_I_10 = scipy.misc.imread('bilder_v27/messung_1_rot_sigma/b_w/m1_I_10_cut-LAB.png', flatten=True, mode=None)
plt.plot(range(0, 4000), rot_I_0[450], 'r-', linewidth = 0.5, label = '$I = 0$')
#plt.plot(range(0, 4000), rot_I_7_5[450], 'b-', linewidth = 0.5)
plt.plot(range(0, 4000), rot_I_10[450], 'b-', linewidth = 0.5, label = '$I = 10$A')
plt.xlim(0, 4000)
plt.ylim(60, 160)
plt.xlabel('$x$/px')
plt.ylabel('Helligkeit')
plt.legend()
plt.grid()
plt.savefig('plots/rot_sigma_intensitaet.pdf')
#plt.show()


plt.clf()

plt.subplot(211)
plt.plot(range(0, 4000), rot_I_0[450], 'r-', linewidth = 0.5)
peaks_rot_0 = np.genfromtxt('data/peaks_rot_sigma_I_0.txt', unpack = True)
for i in peaks_rot_0:
    plt.axvline(x = i, linewidth=0.7, color='k')
plt.ylabel('Helligkeit')
plt.xlabel('$x$/px')
plt.xlim(0, 4000)

plt.subplot(212)
img = mpimg.imread('bilder_v27/messung_1_rot_sigma/m1_I_0_cut.png')
plt.imshow(img)
for i in peaks_rot_0:
    plt.axvline(x = i, linewidth=1.2, color='w')
plt.xlabel('$x$/px')
plt.savefig('plots/peaks_rot_sigma_0.pdf')

plt.clf()
plt.subplot(211)
plt.plot(range(0, 4000), rot_I_10[450], 'r-', linewidth = 0.5)
peaks_rot_10 = np.genfromtxt('data/peaks_rot_sigma_I_10.txt', unpack = True)
for i in peaks_rot_10:
    plt.axvline(x = i, linewidth=0.7, color='k')
plt.xlim(0, 4000)
plt.ylabel('Helligkeit')
plt.xlabel('$x$/px')

plt.subplot(212)
img = mpimg.imread('bilder_v27/messung_1_rot_sigma/m1_I_10_cut.png')
plt.imshow(img)
for i in peaks_rot_10:
    plt.axvline(x = i, linewidth=1.2, color='w')
plt.xlabel('$x$/px')
plt.savefig('plots/peaks_rot_sigma_10.pdf')
