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
	if train_Y[index] == 0:
		train_Y_mod.append(1)
	else:
		train_Y_mod.append(-1)

#print(train_X[0], train_X[1])

C = ['0.00001', '0.001', '0.1', '10', '1000']
w_len_array = []
C_size_array = [-5, -3, -1, 1, 3]
#C_size_array = [-5, -3, -1, 1]
for C_index in range(len(C_size_array)):
	mySVM = svm_train(train_Y_mod, train_X, '-t 0 -h 0 -c '+C[C_index])
	support_vectors = mySVM.get_SV()
	coefficients = mySVM.get_sv_coef()
	support_vectors_N = len(support_vectors)
	#print(N, support_vectors_N)
	#print(support_vectors[0], support_vectors[1])
	#print(coefficients[0][0])
	w = [0, 0]
	for index in range(support_vectors_N):
		w[0] += support_vectors[index][1]*coefficients[index][0]
		w[1] += support_vectors[index][2]*coefficients[index][0]
	w_len = math.sqrt(w[0]*w[0]+w[1]*w[1])
	w_len_array.append(w_len)
	print('finish log(10)(C) = ', C_size_array[C_index], 'w_len is ', w_len)
plt.plot(C_size_array, w_len_array)
plt.xlabel('log(10) (C)')
plt.ylabel('|w|')
plt.savefig('11')