# -*- coding: utf-8 -*-
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
'''print datalen'''
'''
for i in range(len(data_list)):
    print(data_list[i])
'''

index=0
correct_case_n=0
correct_time=0
weigh=[0.0, 0.0, 0.0, 0.0, 0.0]
while correct_case_n != data_size:
    '''for i in range(400):'''
    '''print(data_list[index][5], sign( multiply(weigh, data_list[index]) ))'''
    if data_list[index][5] == sign( multiply(weigh, data_list[index]) ):
        correct_case_n+=1
    else:
        for i in range(5):
            weigh[i]+=data_list[index][5]*data_list[index][i]
        correct_case_n=0
        correct_time+=1
        print ("wrong in index ", index , " and correct time is " , correct_time)
    index+=1
    if index == data_size:
        index=0
print(correct_time)
