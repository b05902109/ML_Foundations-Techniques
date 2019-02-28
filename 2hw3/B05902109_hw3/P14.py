import numpy as np
import matplotlib.pyplot as plt
import math
from operator import itemgetter

trainX=[]
trainY=[]
#testX=[]
#testY=[]

with open("train.txt" , "r") as fptr:
	for line in fptr:
		temp = line.split()
		trainY.append(float(temp[-1]))
		trainX.append([float(temp[0]), float(temp[1])])

'''
with open("test.txt" , "r") as fptr:
	for line in fptr:
		temp = line.split()
		testY.append(float(temp[-1]))
		testX.append([float(temp[0]), float(temp[1])])
'''

def Count_Ein(X, Y, theta, s, index):
	Ein = 0.0
	incorrect = [0] * len(X)
	for i in range(len(X)):
		Ytmp = np.sign(X[i][index] - theta)
		Yhat = s * ( 1 if Ytmp == 0 else Ytmp )			
		if Yhat != Y[i]:
			Ein += 1
			incorrect[i] = 1
	return (Ein/len(X)), incorrect

def DecisionStump(X, Y, U):
	Xlen = len(X)
	dim = len(X[0])
	best_theta = 0
	best_s = 0
	best_Ein_weighted = 1.0
	best_index = 0
	for theta_index in range(Xlen):
		for index in range(dim):
			for s in {-1, 1}:
				theta = X[theta_index][index]
				Ein_weighted = 0.0
				for i in range(Xlen):
					Ytmp = np.sign(X[i][index] - theta)
					Yhat = s * ( 1 if Ytmp == 0 else Ytmp)
					if Yhat != Y[i]:
						Ein_weighted += U[i]
				if Ein_weighted < best_Ein_weighted:
					best_theta = theta
					best_s = s
					best_index = index
					best_Ein_weighted = Ein_weighted
	#print(best_Ein_weighted, best_theta, best_s, best_index)
	return best_theta, best_s, best_Ein_weighted, best_index

def Count_Gt_Ein(T, alpha_array, theta_array, s_array, index_array):
	Ein = 0.0
	for i in range(len(trainX)):
		Yhat = 0.0
		for t in range(T + 1):
			ytmp = np.sign(trainX[i][index_array[t]] - theta_array[t])
			yhat = s_array[t] * (1 if ytmp == 0 else ytmp)
			Yhat += yhat
		if np.sign(Yhat) != trainY[i]:
			Ein += 1.0
	return Ein/len(trainX)

def AdaBoost(X, Y, T):
	Xlen = len(X)
	dim = len(X[0])
	U = [1.0 / Xlen] * Xlen
	alpha = np.zeros((T,))
	#Ein_array = []
	#Ein_Gt_array = []
	#theta_array = []
	#s_array = []
	#index_array = []
	U_array = []
	#adaBoost
	for t in range(T):
		print(t)
		theta, s, Ein_weighted, index = DecisionStump(X, Y, U)
		#theta_array.append(theta)
		#s_array.append(s)
		#index_array.append(index)
		#print(Ein_weighted)
		Ein, incorrect = Count_Ein(X, Y, theta, s, index)
		#Ein_array.append(Ein)
		epslon = Ein_weighted / np.sum(U)
		U_array.append(np.sum(U))
		err_t = np.sqrt(( 1 - epslon ) / epslon)
		#print(epslon)
		alpha[t] = np.log(err_t)
		#Ein_Gt_array.append(Count_Gt_Ein(t, alpha, theta_array, s_array, index_array))
		for i in range(Xlen):
			if incorrect[i]:
				U[i] *= err_t
			else:
				U[i] /= err_t
	return U_array

#print(X_sort[0])
U_array = AdaBoost(trainX, trainY, 300)
print('U_2 = %f'%(U_array[1]))
print('U_300 = %f'%(U_array[300-1]))		
plt.plot(U_array)
plt.xlabel('t')
plt.ylabel('U_t')
plt.show()