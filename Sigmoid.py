import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

probs = np.arange(0,1,0.01)
odds = [p/(1-p) for p in probs]
plt.plot(probs,odds)
plt.show()