# -*-coding:utf-8 -*-
# -on my iMac-

# @File      : Matplotlib操作.py
# @Time      : 2022/2/3 下午7:13
# @Author    : Ronchy
import numpy as np
import matplotlib.pyplot as plt
def func(x):
    return np.square(x)

def dfunc(x):
    return 2*x

def gradient_descent(x_start,func_deri,epochs,learning_rate):
    theta_x = np.zeros(epochs+1)
    temp_x = x_start
    theta_x[0] = temp_x
    for i in range(epochs):
        deri_x = func_deri(temp_x)
        delta = -deri_x*learning_rate
        temp_x = temp_x+delta
        theta_x[i+1] = temp_x
    return theta_x

def mat_plot():
    line_x = np.linspace(-5,5,100)
    line_y = func(line_x)
    x_start = -5
    epochs = 5
    lr = 0.3
    x = gradient_descent(x_start,dfunc,epochs,lr)

    color = 'r'
    plt.plot(line_x,line_y,c='b')
    plt.scatter(x,func(x),c=color,label='lr={}'.format(lr))
    plt.scatter(x,func(x),c= color)
    plt.legend()
    plt.show()

mat_plot()