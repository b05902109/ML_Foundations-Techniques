import numpy as np
import matplotlib.pyplot as plt

def dataset(M):
    #count = 0
    xofX=np.random.uniform(-1,1,M)
    yofX=np.random.uniform(-1,1,M)
    X=np.c_[np.ones(M), xofX,yofX, np.ones(M), np.ones(M), np.ones(M)]
    for i in range(M):
        X[i, 3]=X[i, 1]*X[i, 2]
        X[i, 4]=X[i, 1]*X[i, 1]
        X[i, 5]=X[i, 2]*X[i, 2]
    Y=np.sign(X[:,1]*X[:,1]+X[:,2]*X[:,2]-0.6)
    for i in range(M):
        if np.random.rand() <= 0.1:
            #count += 1
            Y[i] *= -1
    Y = np.mat(Y)
    Y = np.reshape(Y, (M, 1))
    #print(count)
    #print(X, Y)
    return X, Y

M = 1000
#dataset(M)
w_total=np.reshape(np.mat(np.ones(6)), (6,1))
E_out = 0.0
E_out_array = [ ]
for i in range(1000):
    X, Y = dataset(M)
    Xtest, Ytest = dataset(M)
    #w = np.linalg.pinv(X).dot(Y)
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    w_total += w
    Y_out = np.sign(Xtest.dot(w))
    error = np.sum(Y_out!=Ytest)/M
    E_out += error
    E_out_array.append(error)
    #print(error)
print ('avg weight : ', w_total[0,0]/1000, w_total[1,0]/1000, w_total[2,0]/1000, w_total[3,0]/1000, w_total[4,0]/1000, w_total[5,0]/1000)
print ('avg E_out : ', E_out/1000)
plt.hist(E_out_array, bins=np.arange(0, 0.2, 0.001))
plt.show()
