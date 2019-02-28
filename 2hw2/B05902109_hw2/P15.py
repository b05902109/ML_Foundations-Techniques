import numpy as np
import math
import random

fptr = open("data.txt" , "r")
train_X = []
train_Y = []
test_X = []
test_Y = []
temp_list=[ ]
count = 0
for line in fptr:
    if count < 400:
        temp_list = ([float(x) for x in line.split()])
        temp_list.insert(0, float(1))
        train_Y.append(float(temp_list[-1]))
        temp_list.pop();
        train_X.append(temp_list)
        count += 1
    else:
        temp_list = ([float(x) for x in line.split()])
        temp_list.insert(0, float(1))
        test_Y.append(float(temp_list[-1]))
        temp_list.pop();
        test_X.append(temp_list)
        count += 1
fptr.close()
train_n = len(train_X)
test_n = len(test_X)
dimension = len(train_X[0])
train_X_m = np.mat(train_X)
#train_Y_m = np.mat(train_Y)
#test_X_m = np.mat(test_X)
#test_Y_m = np.mat(test_Y)


lamda_array = [0.01, 0.1, 1, 10, 100]
E_in = [0.0]*5
train_Y_predict = [0]*5
for i in range(5):
    train_Y_predict[i] = [0]*train_n

for iteration in range(250):
    #print('iteration = ', iteration)
    bootstapped_setX = []
    bootstapped_setY = []
    for i in range(400):
        bootstapped = random.randint(0, train_n-1)
        #print(bootstapped)
        bootstapped_setX.append(train_X[bootstapped])
        bootstapped_setY.append(train_Y[bootstapped])
    bootstapped_setX_m = np.mat(bootstapped_setX)
    bootstapped_setY_m = np.mat(bootstapped_setY)
    for lamda_index in range(5):
        w = np.linalg.inv(lamda_array[lamda_index]*np.eye(dimension)+bootstapped_setX_m.T.dot(bootstapped_setX_m)).dot(bootstapped_setX_m.T).dot(bootstapped_setY_m.T)
        predict = np.sign(train_X_m.dot(w))
        for i in range(train_n):
            train_Y_predict[lamda_index][i] += predict[i]
#print(train_Y_predict)
for i in range(5):
    for j in range(train_n):
        if  np.sign(train_Y_predict[i][j]) != train_Y[j] :
            E_in[i] += 1.0
    E_in[i] /= train_n
    print('lamda = %g\t, E_in = %g'%(lamda_array[i], E_in[i]))
