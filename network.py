import numpy as np
import pandas as plt
#we use softmax here that is ei/(sigma ei)
import matplotlib.pyplot as plot 
#maths 
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
    one=[np.arrange(y.size),y]=1
    return one.T
def backprop(w1,a1,w2,a2,x,y):
    m=len(y)
    one=onehot(y)
    dz2=a2-one
    dw2=(1/m)*(dz2.dot(a1.T))#hidden layer to output layer
    db2=(1/m)*(np.sum(dz2))
    dz1=a1-one
    dw1=(1/m)*(dz1.dot(x.T)) # input layer to hidden layer
    db1=(1/m)*(np.sum(dz1))
    return dw1,dw2,db1.db2
def updte(w1,b1,w2,b2,dw1,dw2,db1,db2,lr):
    w1-=lr*dw1
    b1-=lr*db1
    w2-=lr*dw2
    b2-=lr*db2
    