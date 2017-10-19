#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import numpy as np
#
#img = mpimg.imread('m2_I_0_grey.png')
#
#plt.plot(range(1000), img[1500][:][[range(1000)], 0][0])
#print(img)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# generate some sample data
import scipy.misc
lena = scipy.misc.imread('m2_I_0_grey.png', flatten=True, mode=None)
plt.plot(range(0, 4000)[1000: 3250], lena[300][1000: 3250])
#print(lena[250])
plt.show()
#lena = scipy.misc.imresize(lena, 0.15, interp='cubic')

# create the x and y coordinate arrays (here we just use pixel indices)
xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

# create the figure
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.gray,
#        linewidth=0)

# show it
#plt.show()
