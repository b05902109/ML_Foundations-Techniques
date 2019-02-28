import numpy as np
import matplotlib.pyplot as plt
import math
from operator import itemgetter

trainX=[]
trainY=[]
testX=[]
testY=[]

with open("train.txt" , "r") as fptr:
	for line in fptr:
		temp = line.split()
		trainY.append(float(temp[-1]))
		trainX.append([float(temp[0]), float(temp[1])])


with open("test.txt" , "r") as fptr:
	for line in fptr:
		temp = line.split()
		testY.append(float(temp[-1]))
		testX.append([float(temp[0]), float(temp[1])])


def Count_Ein(X, Y, theta, s, index):
	incorrect = [0] * len(X)
	for i in range(len(X)):
		Ytmp = np.sign(X[i][index] - theta)
		Yhat = s * ( 1 if Ytmp == 0 else Ytmp )			
		if Yhat != Y[i]:
			incorrect[i] = 1
	return incorrect

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

def Count_Gt_Eout(T, alpha_array, theta_array, s_array, index_array):
	Eout = 0.0
	for i in range(len(testX)):
		Yhat = 0.0
		for t in range(T + 1):
			ytmp = np.sign(testX[i][index_array[t]] - theta_array[t])
			yhat = s_array[t] * (1 if ytmp == 0 else ytmp)
			Yhat += yhat * alpha_array[t]
		if np.sign(Yhat) != testY[i]:
			Eout += 1
	return Eout/len(testX)

def AdaBoost(X, Y, T):
	Xlen = len(X)
	dim = len(X[0])
	U = [1.0 / Xlen] * Xlen
	alpha = np.zeros((T,))
	#Ein_array = []
	Eout_Gt_array = []
	theta_array = []
	s_array = []
	index_array = []
	#adaBoost
	for t in range(T):
		print(t)
		theta, s, Ein_weighted, index = DecisionStump(X, Y, U)
		theta_array.append(theta)
		s_array.append(s)
		index_array.append(index)
		incorrect = Count_Ein(X, Y, theta, s, index)
		epslon = Ein_weighted / np.sum(U)
		#print(epslon, theta, s, index)
		err_t = np.sqrt(( 1 - epslon ) / epslon)
		alpha[t] = np.log(err_t)
		Eout_Gt_array.append(Count_Gt_Eout(t, alpha, theta_array, s_array, index_array))
		for i in range(Xlen):
			if incorrect[i]:
				U[i] *= err_t
			else:
				U[i] /= err_t
	return Eout_Gt_array

#print(X_sort[0])
Eout_array = AdaBoost(trainX, trainY, 300)
print('Eout(G) =', Eout_array[-1])
print(Eout_array[-3], Eout_array[-2], Eout_array[-1])
plt.plot(Eout_array)
plt.xlabel('t')
plt.ylabel('Eout(Gt)')
plt.show()