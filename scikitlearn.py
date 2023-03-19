from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y, test_size=0.2, random_state=42)
np.unique(y_train, return_counts=True)

sgd = SGDClassifier(loss='log_loss', max_iter = 100, tol=1e-3, random_state=42)
sgd.fit(x_train, y_train)
print(sgd.score(x_train,y_train))

sgd.predict(x_test[0:10])