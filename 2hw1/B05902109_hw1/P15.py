import numpy as np
import matplotlib.pyplot as plt
import math
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

fptr = open("2hw1_test.txt" , "r")
test_X=[]
test_Y=[]
for line in fptr:
	temp = line.split()
	test_Y.append(float(temp[0]))
	test_X.append([float(temp[1]), float(temp[2])])
fptr.close()
test_N = len(test_Y)

train_Y_mod = []
for index in range(train_N):
	if train_Y[index] == 0:
		train_Y_mod.append(1)
	else:
		train_Y_mod.append(-1)
test_Y_mod = []
for index in range(test_N):
	if test_Y[index] == 0:
		test_Y_mod.append(1)
	else:
		test_Y_mod.append(-1)

#print(tmp, N)

gamma = ['1', '10', '100', '1000', '10000']
Eout = []
gamma_size_array = [0, 1, 2, 3, 4]
#gamma_size_array = [0, 1]
for gamma_index in range(len(gamma_size_array)):
	mySVM = svm_train(train_Y_mod, train_X, '-t 2 -h 0 -c 0.1 -g '+gamma[gamma_index])
	p_labs, p_acc, p_vals = svm_predict(test_Y_mod, test_X, mySVM)
	Eout.append((100-float(p_acc[0]))/100)

plt.plot(gamma_size_array, Eout)
plt.xlabel('log(10) (C)')
plt.ylabel('Eout')
plt.savefig('15')
