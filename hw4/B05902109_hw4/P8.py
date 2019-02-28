import numpy as np
import matplotlib.pyplot as plt
import math

fptr = open("train.txt" , "r")
trainX = [ ]
trainY=[ ]
temp_list=[ ]
for line in fptr:
	temp_list=([float(x) for x in line.split()])
	trainY.append(temp_list[-1])
	temp_list.pop()
	temp_list.insert(0, float(1))
	trainX.append(temp_list)
train_size = len(trainX)
print(train_size)

fptr = open("test.txt" , "r")
testX = [ ]
testY=[ ]
temp_list=[ ]
for line in fptr:
	temp_list=([float(x) for x in line.split()])
	testY.append(temp_list[-1])
	temp_list.pop()
	temp_list.insert(0, float(1))
	testX.append(temp_list)
test_size = len(testX)
fptr.close()
print(test_size)

def reg_lin_regression(in_lamba):
	return np.linalg.inv(np.mat(trainX[0:120]).T.dot(np.mat(trainX[0:120]))+in_lamba*np.identity(len(trainX[1]))).dot(np.mat(trainX[0:120]).T).dot(np.mat(trainY[0:120]).T)

def check_train(w):
	err = 0.0
	for i in range(120):
		if np.sign(np.mat(trainX[i]).dot(w)) != np.sign(trainY[i]):
			err = err + 1
	#print(err/train_size)
	return err/120

def check_val(w):
	err = 0.0
	for i in range(train_size-120):
		if np.sign(np.mat(trainX[120+i]).dot(w)) != np.sign(trainY[120+i]):
			err = err + 1
	#print(err/test_size)
	return err/(train_size-120)

lamda_list = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2]

Etrain = []
Eval = []
for lamda in lamda_list:
	w_reg = reg_lin_regression(10**lamda)
	print(w_reg)
	Etrain.append(check_train(w_reg))
	Eval.append(check_val(w_reg))

plt.plot(lamda_list, Etrain, label='E_train')
plt.plot(lamda_list, Eval, label='E_val')
plt.legend(loc='upper left')
plt.xlabel('log_10 lamda')
plt.show()