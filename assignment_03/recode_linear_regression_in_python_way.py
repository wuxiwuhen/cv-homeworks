# Linear Regression

"""
recode linear regression in python's way by wuxi

"""
import numpy as np
import random
from matplotlib import pyplot as plt
import os

def inference(w, b, x):        # inference, test, predict, same thing. Run model after training
    pred_y = w * np.array(x) + b
    return pred_y

def eval_loss(w, b, x_list, gt_y_list):
    avg_loss = sum(0.5*(np.array(x_list) * w + b - np.array(gt_y_list))**2)/len(gt_y_list)
    return avg_loss

def gradient(pred_y, gt_y, x):
    diff = np.array(pred_y) - np.array(gt_y)
    dw = diff * np.array(x)
    db = diff
    return dw, db

def cal_step_gradient(batch_x_list, batch_gt_y_list, w, b, lr):
    batch_size = len(batch_x_list)
    pred_y = inference(w, b, batch_x_list)
    dw, db = gradient(pred_y, batch_gt_y_list, batch_x_list)
    avg_dw = sum(dw)/batch_size
    avg_db = sum(db)/batch_size
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x_list, gt_y_list, batch_size, lr, max_iter):
    w = 0
    b = 0
    loss = []
    for _ in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size)
        batch_x = np.array(x_list)[batch_idxs]
        batch_y = np.array(gt_y_list)[batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        loss.append(eval_loss(w, b, x_list, gt_y_list))
    print('w:{0}, b:{1},w/b:{2}'.format(w, b, w/b))
    print('loss is {0}'.format(eval_loss(w, b, x_list, gt_y_list)))
    return w, b, np.array(loss)

def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = np.random.randint(0,100,num_samples) * np.random.random(num_samples)
    y_list = x_list * w + b + np.random.random(num_samples) * np.random.randint(-5, 6, num_samples)
    return x_list, y_list, w, b

def run():
    x_list, y_list, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 50
    w1, b1, loss = train(x_list, y_list, 50, lr, max_iter)
    fig, ax = plt.subplots(2)
    ax[0].scatter(x_list, y_list, marker= 'o',c = 'r')
    ax[0].plot(np.linspace(0,99,100), np.linspace(0,99,100) * w1 + b1, c = 'b')
    ax[0].set_title("scatters and the final fitting line")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[1].plot(np.linspace(1,max_iter,max_iter),loss)
    ax[1].set_title("loss value for every step of iteration")
    ax[1].set_xlabel("iter_step")
    ax[1].set_ylabel("loss")
    plt.tight_layout()
    plt.show()
    

if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()