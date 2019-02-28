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

tmp = 0
train_Y_mod = []
for index in range(N):
	if train_Y[index] == 0:
		train_Y_mod.append(1)
		tmp += 1
	else:
		train_Y_mod.append(-1)

#print(tmp, N)

C = ['0.001', '0.01', '0.1', '1', '10']
dis = []
C_size_array = [-3, -2, -1, 0, 1]
#C_size_array = [-3]
for C_index in range(len(C_size_array)):
	C_now = float(C[C_index])
	mySVM = svm_train(train_Y_mod, train_X, '-t 2 -h 0 -g 80 -c '+C[C_index])
	coefficients = mySVM.get_sv_coef()
	support_vectors = mySVM.get_SV()
	p_labs, p_acc, p_vals = svm_predict(train_Y_mod, train_X, mySVM)
	#print('len(coefficients) = ', len(coefficients), 'len(p_vals) = ', len(p_vals))
	#print(coefficients)
	i = 0
	while not(0 < abs(float(coefficients[i][0])) < C_now):
		#print(abs(float(coefficients[i][0])))
		i += 1
	j = 0
	while not(train_X[j][0] == support_vectors[i][1] and train_X[j][1] == support_vectors[i][2]):
		j += 1
	dis.append(abs(float(p_vals[j][0])))
	#print('dis =', p_vals[j])
	#print('support_vectors = ', support_vectors[i], 'coefficients = ', coefficients[i])
	#print('train_X = ', train_X[j], 'train_Y = ', train_Y[j])
	'''
	for i in range(len(p_vals)):
		if
	'''

print(dis)
plt.plot(C_size_array, dis)
plt.xlabel('log(10) (C)')
plt.ylabel('dis')
plt.ylim(0.9, 1.1)
plt.savefig('14')
