import numpy as np
import pandas as pd
#we use softmax here that is ei/(sigma ei)
import matplotlib.pyplot as plot 
#maths 
data = pd.read_csv('mnist_test.csv')
data=np.array(data)
m,n=data.shape
np.random.shuffle(data)
data_train=data.T
y=data_train[0]
x=data_train[1:n]
x=x/255
def inpar():
    w1=np.random.rand(10,784)-0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def relu(z):
    return np.maximum(0,z)
def softmax(z):
    return np.exp(z)/sum(np.exp(z))
def for_prp(w1,w2,b1,b2,x):
    z1=np.dot(w1,x)+b1
    a1=relu(z1)
    z2=np.dot(w2,x)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2
def rder(z): # relu derivative
    return z>0 
def onehot(y):
    one=np.zeros((y.size,y.max()+1))
    one[np.arrange(y.size),y]=1
    return one.T
def backprop(z1,z2,w1,a1,w2,a2,x,y):
    m=len(y)
    one=onehot(y)
    dz2=a2-one
    dw2=(1/m)*(dz2.dot(a1.T))#hidden layer to output layer
    db2=(1/m)*(np.sum(dz2))
    dz1=np.dot(w2.T,dz2)*rder(z1)
    dw1=(1/m)*(dz1.dot(a1.T)) # input layer to hidden layer
    db1=(1/m)*(np.sum(dz1,axis=1,keepdims=True))
    return dw1,dw2,db1,db2
def updte(w1,b1,w2,b2,dw1,dw2,db1,db2,lr):
    w1-=lr*dw1
    b1-=lr*db1
    w2-=lr*dw2
    b2-=lr*db2
    return w1,b1,w2,b2
def pred(a2):
    return np.argmax(a2,0)
def accuarcy(preds,y):
    print(preds, y)
    return np.sum(preds == y) / y.size
def gradedes(x,y,lr,epochs):
    w1,b1,w2,b2=inpar()
    for i in range (epochs):
        z1,a1,z2,a2=for_prp(w1,w2,b1,b2,x)
        dw1,dw2,db1,db2=backprop(z1,z2,w1,a1,w2,a2,x,y)
        w1,b1,w2,b2=updte(w1,b1,w2,b2,dw1,dw2,db1,db2,lr)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = pred(a2)
            print(accuarcy(predictions, y))
    return w1, b1, w2, b2
