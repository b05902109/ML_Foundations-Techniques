import numpy as np
import math

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
train_X_m = np.mat(train_X)
train_Y_m = np.mat(train_Y)
test_X_m = np.mat(test_X)
test_Y_m = np.mat(test_Y)

dimension = len(train_X[0])

lamda_array = [0.01, 0.1, 1, 10, 100]

for lamda in lamda_array:
    w = np.linalg.inv(lamda*np.eye(dimension)+train_X_m.T.dot(train_X_m)).dot(train_X_m.T).dot(train_Y_m.T)
    train_Y_predict = train_X_m.dot(w)
    E_in = 0.0
    for i in range(train_n):
        if np.sign(train_Y_predict[i]) != train_Y[i]:
            E_in += 1.0
    E_in /= train_n
    print('lamda = %-g\t, E_in = %-g'%(lamda, E_in))