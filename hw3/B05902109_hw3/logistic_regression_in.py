from __future__ import division, print_function
from visual import *
from random import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt
size=21
N=1000
time=2000
ita=0.001
X=np.zeros(shape=(N, size))
y=np.zeros(shape=(N))
w1=np.zeros(shape=(size))
w2=np.zeros(shape=(size))
GD=np.zeros(shape=(size))
def sign(a):
	return 1 if a>=0 else -1
for i in range(N):
	X[i][0]=1
	tmp=raw_input().split()
	for j in range(size-1):
		X[i][j+1]=tmp[j]
	y[i]=tmp[size-1]
E_in_GD=[]
E_in_SGD=[]
Time=[]
for t in range(time):
	#GD
	for i in range(size):
		GD[i]=0
	for i in range(N):
		theta=1/(1+math.exp(y[i]*np.dot(X[i], w1)))
		GD+=theta*(-y[i]*X[i])
	GD/=N
	w1-=ita*GD

	#SGD
	theta=1/(1+math.exp(y[t%N]*np.dot(X[t%N], w2)))
	GD=theta*(y[t%N]*X[t%N])
	w2+=ita*GD

	error1=0.0
	error2=0.0
	for i in range(N):
		if sign(np.dot(X[i], w1)) != y[i]:
			error1+=1
		if sign(np.dot(X[i], w2)) != y[i]:
			error2+=1
	E_in_GD.append(error1/N)
	E_in_SGD.append(error2/N)
	Time.append(t+1)
plt.title('lr=0.001')
plt.xlabel("t")
plt.ylabel("E_in")
plt.plot(Time, E_in_GD, alpha=1, label='Gradient Descent')
plt.plot(Time, E_in_SGD, alpha=0.5, label='Stochastic Gradient Descent')
plt.legend(loc='upper right', shadow=True)
plt.show()