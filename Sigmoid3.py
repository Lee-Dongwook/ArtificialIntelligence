import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

zs = np.arange(-10., 10., 0.1)
gs = [1/(1+np.exp(-z)) for z in zs]
plt.plot(zs, gs)
plt.show()
