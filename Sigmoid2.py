import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

probs = np.arange(0.001,0.999,0.001)
logit = [np.log(p/(1-p)) for p in probs]
plt.plot(probs, logit)
plt.show()