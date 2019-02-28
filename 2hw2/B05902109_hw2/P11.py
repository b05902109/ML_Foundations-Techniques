import numpy as np
import math

fptr = open("data.txt" , "r")
train_X = []
train_Y = []
test_X = []
test_Y = []
temp_list=[ ]
count = 0
for line in fptr:
    if count < 400:
        temp_list = ([float(x) for x in line.split()])
        #temp_list.insert(0, float(1))
        train_Y.append(float(temp_list[-1]))
        temp_list.pop();
        train_X.append(temp_list)
        count += 1
    else:
        temp_list = ([float(x) for x in line.split()])
        #temp_list.insert(0, float(1))
        test_Y.append(float(temp_list[-1]))
        temp_list.pop();
        test_X.append(temp_list)
        count += 1
fptr.close()
train_n = len(train_X)
test_n = len(test_X)
train_X_m = np.mat(train_X)
train_Y_m = np.mat(train_Y)
test_X_m = np.mat(test_X)
test_Y_m = np.mat(test_Y)

dimension = len(train_X[0])

def create_kernel(X1, X2, gamma):
    row1, column1 = X1.shape
    row2, column2 = X2.shape
    kernel = np.zeros((row1, row2))
    for i in range(row1):
        for j in range(row2):
            kernel[i, j] = math.exp(-1.0*gamma*(X1[i]-X2[j]).dot((X1[i]-X2[j]).T))
    return kernel

gamma_array = [32, 2, 0.125]
lamda_array = [0.001, 1, 1000]

for gamma in gamma_array:
    #print(train_X_m.shape)
    Ktrain = create_kernel(train_X_m, train_X_m, gamma)
    #Ktest = create_kernel(train_X_m, test_X_m, gamma)
    #print(Ktrain.shape)
    for lamda in lamda_array:
        beta = np.linalg.inv(lamda*np.eye(Ktrain.shape[0])+Ktrain).dot(train_Y_m.T)
        train_Y_predict = Ktrain.T.dot(beta)
        E_in = 0.0
        for i in range(train_n):
            if np.sign(train_Y_predict[i]) != train_Y[i]:
                E_in += 1.0
        E_in /= train_n
        print('gamma = %-g\t, lamda = %-g\t, E_in = %-g'%(gamma, lamda, E_in))