from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
x = cancer.data
y = cancer.target 

x_train_all , x_test, y_train_all, y_test = train_test_split(x,y,stratify =y, test_size=0.2 ,random_state=42)

sgd = SGDClassifier(loss='hinge', random_state=42)
sgd.fit(x_train_all, y_train_all)

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

class SingleLayer:
    def __init__(self, learning_rate=0.1, l1=0, l2=0):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
    
    def forpass(self,x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None)
        a = 1/(1+np.exp(-z)) 
        return a
    
    def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.backprop(x[i], err)
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                self.w -= self.lr * w_grad
                self.b -= self.lr * b_grad
                self.w_history.append(self.w.copy())
                a=np.clip(a,1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y)+self.reg_loss())
            self.update_val_loss(x_val, y_val)
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) >= 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w))+ self.l2 / 2* np.sum(self.w**2)
    
    def update_val_loss(self,x_val,y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a,1e-10,1-1e-10)
            val_loss += -(y_val[i]*np.log(a)+(1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val)+self.reg_loss())
    

layer1 = SingleLayer()
layer1.fit(x_train,y_train)
print(layer1.score(x_val, y_val))

w2 = []
w3 = []

for w in layer1.w_history:
    w2.append(w[2])
    w3.append(w[3])

# plt.plot(w2,w3)
# plt.plot(w2[-1],w3[-1],'ro')
# plt.plot(w2[-5],w3[-5],'ro')

train_mean = np.mean(x_train, axis = 0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train-train_mean) / train_std

val_mean = np.mean(x_val, axis=0)
val_std = np.std(x_val, axis=0)
x_val_scaled = (x_val - val_mean) / val_std

layer2 = SingleLayer()
layer2.fit(x_train_scaled, y_train)
print(layer2.score(x_val_scaled, y_val))

#과대 적합 , 과소 적합

# layer3 = SingleLayer()
# layer3.fit(x_train_scaled, y_train, x_val = x_val_scaled, y_val = y_val)
# plt.ylim(0, 0.3)
# plt.xlim(0,100)
# plt.plot(layer3.losses)
# plt.plot(layer3.val_losses)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train_loss','val_loss'])
# plt.show()


# 규제
# l1_list = [0.0001, 0.001, 0.01]
# for l1 in l1_list:
#     lyr = SingleLayer(l1 = l1)
#     lyr.fit(x_train_scaled, y_train, x_val = x_val_scaled, y_val=y_val)
    
#     plt.plot(lyr.losses)
#     plt.plot(lyr.val_losses)
#     plt.title('Learning Curve (l1={})'.format(l1))
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train_loss','val_loss'])
#     plt.ylim(0,0.3)
#     plt.xlim(0,100)
#     plt.show()


# 교차 검증
validation_scores = []
k = 10
bins = len(x_train_all) // k

for i in range(k):
    start = i*bins
    end = (i+1)*bins
    val_fold = x_train_all[start:end]
    val_target = y_train_all[start:end]
    
    train_index = list(range(0,start)) + list(range(end, len(x_train_all)))
    train_fold = x_train_all[train_index]
    train_target = y_train_all[train_index]
    
    train_mean = np.mean(train_fold, axis=0)
    train_std = np.std(train_fold, axis=0)
    train_fold_scaled = (train_fold - train_mean) / train_std
    val_fold_scaled = (val_fold - train_mean) / train_std
    
    lyr = SingleLayer(l2 = 0.01)
    lyr.fit(train_fold_scaled, train_target, epochs = 50)
    score = lyr.score(val_fold_scaled, val_target)
    validation_scores.append(score)
print(np.mean(validation_scores))