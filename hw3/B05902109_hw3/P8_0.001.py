import numpy as np
import matplotlib.pyplot as plt
import math

fptr = open("hw3_train.txt" , "r")
data_train_X=[ ]
data_train_Y=[ ]
temp_list=[ ]
for line in fptr:
    temp_list=([float(x) for x in line.split()])
    #temp_list.insert(0, float(1))
    data_train_Y.append(float(temp_list[-1]))
    temp_list.pop();
    data_train_X.append(temp_list)
fptr.close()
data_train_size=len(data_train_X)
dimension = len(data_train_X[0])

fptr = open("hw3_test.txt" , "r")
data_test_X=[ ]
data_test_Y=[ ]
for line in fptr:
    temp_list=([float(x) for x in line.split()])
    #temp_list.insert(0, float(1))
    data_test_Y.append(float(temp_list[-1]))
    temp_list.pop();
    data_test_X.append(temp_list)
fptr.close()
data_test_size=len(data_test_X)


def check_Ein(w):
    err = 0
    for index1 in range(data_train_size):
        if data_train_Y[index1] != np.sign(np.dot(np.array(data_train_X[index1]), w)):
            err += 1
    return err/data_train_size

def check_Eout(w):
    err = 0
    for index1 in range(data_test_size):
        if data_test_Y[index1] != np.sign(np.dot(np.array(data_test_X[index1]), w)):
            err += 1
    return err/data_test_size

def sigmoid(X):
    return 1.0/(1.0+math.exp(X))

def GD(w):
    gradient = np.array([0.0]*dimension)
    for index in range(data_train_size):
        gradient += sigmoid( data_train_Y[index]*(np.dot(data_train_X[index], w)) )*-data_train_Y[index]*np.array(data_train_X[index])
    return gradient/data_train_size

def SGD(w, t):
    index = t%data_train_size
    gradient = sigmoid( data_train_Y[index]*(np.dot(data_train_X[index], w)) )*-data_train_Y[index]*np.array(data_train_X[index])
    return gradient

T = 2000
ita = 0.001
Ein_array1 = [ ]
Ein_array2 = [ ]
Ein_array3 = [ ]
Ein_array4 = [ ]
w_GD = np.array([0.0]*dimension)
w_SGD = np.array([0.0]*dimension)
for t in range(T):
    w_GD -= ita * GD(w_GD)
    w_SGD -= ita * SGD(w_SGD, t)
    #print(w_SGD)
    Ein_array1.append(check_Ein(w_GD))
    Ein_array2.append(check_Ein(w_SGD))
    #Ein_array3.append(check_Eout(w_GD))
    #Ein_array4.append(check_Eout(w_SGD))
    print('iter : ', t)
#print(chech_Ein(w_out))
plt.plot(Ein_array1, label='P.19_Ein')
plt.plot(Ein_array2, label='P.20_Ein',linestyle='--')
#plt.plot(Ein_array3, label='P.19_Eout')
#plt.plot(Ein_array4, label='P.20_Eout',linestyle='--')
plt.legend(loc='upper right')
plt.show()
#print(np.shape(np.mat(data_train_X)))
#print(data_train_size, data_test_size)
#print ( data_train[0] )
#print ( np.mat(data_train[0]) )
