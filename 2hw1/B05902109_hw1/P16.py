import matplotlib.pyplot as plt
import random
import numpy as np
from svmutil import *

fptr = open("2hw1_train.txt" , "r")
train_X=[]
train_Y=[]
for line in fptr:
	temp = line.split()
	train_Y.append(float(temp[0]))
	train_X.append([float(temp[1]), float(temp[2])])
fptr.close()
train_N = len(train_Y)

train_Y_mod = []
for i in range(train_N):
	if train_Y[i] == 0:
		train_Y_mod.append(1)
	else:
		train_Y_mod.append(-1)


gamma = ['0.1', '1', '10', '100', '1000']
number = []
gamma_size_array = [-1, 0, 1, 2, 3]
#gamma_size_array = [-1, 0]


for count in range(100):
	index = []
	for i in range(train_N):
		index.append(i)
	random.shuffle(index)

	D_X = []
	D_Y = []
	D_X_val = []
	D_Y_val = []
	for i in range(train_N):
		if i < 1000:
			D_X_val.append(train_X[index[i]])
			D_Y_val.append(train_Y_mod[index[i]])
		else:
			D_X.append(train_X[index[i]])
			D_Y.append(train_Y_mod[index[i]])

	min_gamma_index = 0
	min_Eval = 1.0
	for gamma_index in range(len(gamma_size_array)):
		mySVM = svm_train(D_Y, D_X, '-t 2 -h 0 -c 0.1 -g ' + gamma[gamma_index])
		p_labs, p_acc, p_vals = svm_predict(D_Y_val, D_X_val, mySVM)
		if (100-float(p_acc[0]))/100 < min_Eval:
			min_gamma_index = gamma_index
			min_Eval = (100-float(p_acc[0]))/100
	number.append(gamma_size_array[min_gamma_index])

print(number)
plt.hist(number)
plt.xlim((-1, 3))
plt.xlabel('log(10) (gamma)')
plt.ylabel('number')
plt.savefig('16')
