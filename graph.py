import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("output.txt", delimiter=' ')
c = a.T[0]
F = a.T[1]

delta = 12.0e-6-c

plt.plot(delta, F)

plt.show()
