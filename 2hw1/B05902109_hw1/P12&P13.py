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
N = len(train_Y)

train_Y_mod = []
for index in range(N):
	if train_Y[index] == 8:
		train_Y_mod.append(1)
	else:
		train_Y_mod.append(-1)

#print(train_X[0], train_X[1])

C = ['0.00001', '0.001', '0.1', '10', '1000']
nSV = []
Ein = []
C_size_array = [-5, -3, -1, 1, 3]
#C_size_array = [-5, -3, -1]
for C_index in range(len(C_size_array)):
	mySVM = svm_train(train_Y_mod, train_X, '-t 1 -h 0 -g 1 -d 2 -r 1 -c '+C[C_index])
	nSV.append(mySVM.get_nr_sv())
	p_labs, p_acc, p_vals = svm_predict(train_Y_mod, train_X, mySVM)
	Ein.append((100-float(p_acc[0]))/100)
	#print('finish log(10)(C) = ', C_size_array[C_index], ', Ein = ', (100-float(p_acc[0]))/100, p_acc[0])
#print(C_size_array)
#print(Ein)
plt.plot(C_size_array, Ein)
plt.xlabel('log(10) (C)')
plt.ylabel('Ein')
plt.savefig('12')

plt.cla()
plt.plot(C_size_array, nSV)
plt.xlabel('log(10) (C)')
plt.ylabel('nSV')
plt.savefig('13')