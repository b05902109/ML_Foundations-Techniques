# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_data():
    X = [ ]
    Y = [ ]
    for i in range(20):
        X.append(random.uniform(-1, 1))
    X.sort()
    for i in range(20):
        Y.append(np.sign(X[i]))
        if np.random.random()<0.2:
            Y[i] *= -1
    #print(X)
    #print(Y)
    return X, Y

def find_min_Ein(X, Y):
    theta_array = [ ]
    Ein = 20
    theta = 0.0
    s = 0
    for i in range(21):
        if i == 0:
            theta_array.append(float("-inf"))
        elif i == 20:
            theta_array.append(float("inf"))
        else:
            theta_array.append( ( X[i-1]+X[i] ) / 2)
    #print(theta_array)
    for i in range(21):
        #for every theta
        error_left_O = 0
        # +1/-1
        error_left_X = 0
        # -1/+1
        for j in range(20):
            if X[j] < theta_array[i] and Y[j] != 1:
                error_left_O += 1
            if X[j] > theta_array[i] and Y[j] != -1:
                error_left_O += 1
            if X[j] < theta_array[i] and Y[j] != -1:
                error_left_X += 1
            if X[j] > theta_array[i] and Y[j] != 1:
                error_left_X += 1
        #print(error_left_O, error_left_X)
        if error_left_X > error_left_O and Ein > error_left_O:
            Ein = error_left_O
            theta = theta_array[i]
            s = -1
        if error_left_O > error_left_X and Ein > error_left_X:
            Ein = error_left_X
            theta = theta_array[i]
            s = 1
    if theta == float("-inf"):
        theta = -1
    elif theta == float("inf"):
        theta = 1
    #print(Ein)
    return Ein, theta, s
if __name__ == '__main__':
    T = 1000
    Ein_sum = 0.0
    Eout_sum = 0.0
    Ein_array=[ ]
    Eout_array=[ ]
    for i in range(T):
        X, Y = generate_data()
        Ein, theta, s = find_min_Ein(X, Y)
        Eout = 0.5+0.3*s*(abs(theta)-1)
        Ein_sum += Ein
        Eout_sum += Eout
        Ein_array.append(Ein/20.0)
        Eout_array.append(Eout)
    print(Ein_sum/T/20.0)
    print(Eout_sum/T)

    plt.scatter(Ein_array, Eout_array, s=3)
    plt.show()
