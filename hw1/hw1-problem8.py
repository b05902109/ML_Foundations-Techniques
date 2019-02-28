# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
import numpy as np

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

fptr = open("PLA_Data1.txt" , "r")
data_list=[ ]
temp_list=[ ]
for line in fptr:
    temp_list=([float(x) for x in line.split()])
    temp_list.insert(0, float(1))
    data_list.append(temp_list)
fptr.close()
data_size=len(data_list)

all_ctime=[ ]
sum_ctime=0.0
for i in range(2000):
    random_index=list(range(data_size))
    random.shuffle(random_index)
    index=0
    correct_case_n=0
    correct_time=0
    weigh=[0.0, 0.0, 0.0, 0.0, 0.0]
    while correct_case_n != data_size:
        if data_list[random_index[index]][5] == sign( multiply(weigh, data_list[random_index[index]]) ):
            correct_case_n+=1
        else:
            for i in range(5):
                weigh[i]+=data_list[random_index[index]][5]*data_list[random_index[index]][i]
            correct_case_n=0
            correct_time+=1
        index+=1
        if index == data_size:
            index=0
    all_ctime.append(correct_time)
    sum_ctime+=correct_time
print(sum_ctime/2000)
graph=np.array(all_ctime)
plt.hist(graph, bins=80, normed=1)
plt.title("Problem8 Histogram")
plt.ylabel("the frequency of the number")
plt.xlabel("the number of updates")
plt.show()
