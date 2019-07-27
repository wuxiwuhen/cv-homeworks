import numpy as np
import random
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(- np.array(x)))

def inference(theta, batch_X):
    return sigmoid(batch_X.dot(theta))

def eval_loss(theta,h_theta,Y):
    loss = - Y.T.dot(np.log(h_theta)) - (1 - Y.T).dot(np.log(1 - h_theta))
    return loss[0]

def get_batch_data(X,Y,batch_size):
    batch_idxs = np.random.choice(len(X),batch_size)
    batch_X = X[batch_idxs]
    batch_Y = Y[batch_idxs]
    return batch_X, batch_Y

def featureNormalize(x_list):
    mu = np.mean(x_list, axis=0)
    sigma = np.std(x_list, axis=0)
    X = (x_list-mu)/sigma
    return X, mu, sigma

def cal_step_gradient(theta, batch_Y, batch_X, lr):
    h_theta = inference(theta, batch_X)    #batch_size*1
    dt = batch_X.T.dot(h_theta - batch_Y)/len(batch_Y)
    theta -= lr * dt
    return theta 

def train(theta, X, Y, batch_size, lr, max_iter):
    loss = []
    for _ in range(max_iter):
        batch_X, batch_Y = get_batch_data(X,Y,batch_size)
        theta = cal_step_gradient(theta, batch_Y, batch_X, lr)
        h_theta = inference(theta, X)
        loss.append(eval_loss(theta,h_theta,Y))
    return theta, loss

def run():
    data = np.loadtxt('./assignment_03/data/week3_LR.txt', delimiter = ',')
    x_list = data[:,0:2]
    y_list = data[:,2]
    
    label0 = data[data[:,2] == 0]
    label1 = data[data[:,2] == 1]

    (X, mu, sigma) = featureNormalize(x_list) #不做正则化算的结果有问题
    
    m = len(X)
    X = np.column_stack((np.ones([m, 1]),X))
    Y = y_list.reshape(m, 1)
    
    batch_size = 20
    lr = 0.01
    max_iter = 50000
    theta = np.zeros(3).reshape(3,1)
    theta, loss = train(theta, X, Y, batch_size, lr, max_iter)
    
    fig, ax = plt.subplots(2)
    ax[0].scatter(label0[:,0],label0[:,1],marker = '+',c = 'y')
    ax[0].scatter(label1[:,0],label1[:,1],marker = '+',c = 'r')
    pre_y = (-theta[0]-theta[1]*(X[:,1]))*sigma[1]/theta[2]+mu[1]
    ax[0].plot(data[:,0],pre_y)
    ax[1].plot(np.array(range(max_iter)), loss)
    plt.show()
    
if __name__ == '__main__':
    run()