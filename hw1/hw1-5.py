# -*- coding: utf-8 -*-
import random

def multiply(answer, data):
    y=0.0
    for i in range(5):
        y+=answer[i]*data[i]
    return y

def sign(y):
    if y>0:
        return 1.0
    else:
        return -1.0

def errorRate(weigh, data):
    error=0.0
    for i in range(len(data)):
        if data[i][5] != sign( multiply(weigh, data[i]) ):
            error+=1
    return error/len(data)

def pocket(data_train_list, data_test_list):
    random_index=range(len(data_train_list))
    random.shuffle(random_index)
    index=0
    correct_time=0
    weigh=[0.0, 0.0, 0.0, 0.0, 0.0]
    while correct_time<50:
        '''print(data_list[index][5], sign( multiply(weigh, data_list[index]) ))'''
        if data_train_list[random_index[index]][5] != sign( multiply(weigh, data_train_list[random_index[index]]) ):
            for i in range(5):
                weigh[i]+=data_train_list[random_index[index]][5]*data_train_list[random_index[index]][i]
            correct_time+=1
        index+=1
        if index == data_train_size:
            index=0
    return errorRate(weigh, data_test_list)

fptr = open("PLA_Data2_train.txt" , "r")
data_train_list=[ ]
temp_list=[ ]
for line in fptr:
    temp_list=([float(x) for x in line.split()])
    temp_list.insert(0, float(1))
    data_train_list.append(temp_list)
fptr.close()
data_train_size=len(data_train_list)

fptr = open("PLA_Data2_test.txt" , "r")
data_test_list=[ ]
for line in fptr:
    temp_list=([float(x) for x in line.split()])
    temp_list.insert(0, float(1))
    data_test_list.append(temp_list)
fptr.close()
data_test_size=len(data_test_list)

error_rate=0.0
error_sum=0.0
for i in range(2000):
    error_rate=pocket(data_train_list, data_test_list)
    error_sum+=error_rate
    print("time ", i, error_rate)
print(error_sum/2000)
