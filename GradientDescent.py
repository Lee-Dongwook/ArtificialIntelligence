from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
diabetes = load_diabetes()

x = diabetes.data[:, 2]
y = diabetes.target 

w = 1.0
b = 1.0

y_hat = x[0] * w + b
w_inc = w + 0.1

y_hat_inc = w_inc * x[0] + b

w_rate = (y_hat_inc - y_hat) / (w_inc - w)

w_new = w + w_rate

b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc

b_rate = (y_hat_inc - y_hat) / (b_inc - b)

b_new = b + 1

err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err

y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate *err
b_new = b_new + 1 * err

for i in range(1,100):
     for x_i, y_i in zip(x,y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 *err


plt.scatter(x,y)
pt1 = (-0.1, -0.1*w + b)
pt2 = (0.15, 0.15*w + b)
plt.plot([pt1[0], pt2[0]],[pt1[1], pt2[1]])
plt.show()
