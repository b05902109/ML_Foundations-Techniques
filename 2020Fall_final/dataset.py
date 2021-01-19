import pandas as pd
import numpy as np
import os
from math import sqrt

import torch
from torch.utils.data.dataset import Dataset

class DataReader(object):
    """docstring for DataReader"""
    def __init__(self):
        super(DataReader, self).__init__()
        self.mappers = {
            'hotel': {'City Hotel':0, 'Resort Hotel':1},
            'arrival_date_month': {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12},
            # 'arrival_date_month': {'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7, 'September':8, 'October':9, 'November':10, 'December':11},
            'meal': {'BB':0, 'FB':1, 'HB':2, 'SC':3, 'Undefined':4},
            'market_segment': {'Aviation':0, 'Complementary':1, 'Corporate':2, 'Direct':3, 'Groups':4, 'Offline TA/TO':5, 'Online TA':6, 'Undefined':7},
            'distribution_channel': {'Corporate':0, 'Direct':1, 'GDS':2, 'TA/TO':3, 'Undefined':4},
            'reserved_room_type': {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'L':8, 'P':9},
            'assigned_room_type': {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'K':9, 'L':10, 'P':11},
            'deposit_type': {'No Deposit':0, 'Non Refund':1, 'Refundable':2},
            'customer_type': {'Contract':0, 'Group':1, 'Transient':2, 'Transient-Party':3},
        }
        self.mapSize = {0:2,2:13,10:5,11:8,12:5,16:10,17:12,19:3,21:4}

    def refineData(self, data_df):
        print('[LOG] refineData start.')

        # transform text data to number
        for col_name in data_df.columns:
            if col_name in self.mappers:
                data_df[col_name] = data_df[col_name].map(self.mappers[col_name])

        # clean data
        ## clean column
        data_df.drop(['agent', 'company'], axis=1, inplace=True)          # NAN
        data_df.drop(['country'], axis=1, inplace=True)                   # too sparse

        data_df.dropna(inplace=True)

        return data_df

    def getLabels(self, value):
        tmp = value // 10000.0
        return tmp if tmp <= 9.0 else 9.0

    def oneHotNumpy(self, data_np):
        data_np_new = None
        for index in range(data_np.shape[1]):
            # print(index)
            target = data_np[:, index]
            if index in [0,2,10,11,12,16,17,19,21]:
                tmp = np.zeros((target.size, self.mapSize[index]))
                # print(np.arange(target.size))
                # print(target)
                tmp[np.arange(target.size), target.astype(int)] = 1
                if index == 0:
                    data_np_new = tmp
                elif index == 3:
                    data_np_new = np.c_[data_np_new, tmp[:, 1:]]
                else:
                    data_np_new = np.c_[data_np_new, tmp]
            else:
                data_np_new = np.c_[data_np_new, target]
        return data_np_new

            
class TrainDataset(DataReader, Dataset):
    """docstring for TrainDataset"""
    def __init__(self, dataFolder):
        super(TrainDataset, self).__init__()
        self.trainDataPath = None
        self.trainDataLabelPath = None
        
        self.trainDataPath = os.path.join(dataFolder, 'train.csv')
        self.trainDataLabelPath = os.path.join(dataFolder, 'train_label.csv')

        self.X, self.Y, self.trainDataReservation = None, None, None
        self.labels = None

        # self.X                    = numpy[ ... ]                     , size(91510, 24)
        # self.Y                    = numpy[is_canceled, adr]          , size(91510, 2)
        # self.trainDataReservation = pd[arrival_date, stays_in_nights], size(91510, 2)
        # self.labels               = pd[arrival_date, label]          , size(640, 2)

    def initial(self, df, Y, reservation, labels):
        self.X = df
        self.Y = Y
        self.trainDataReservation = reservation
        self.labels = labels

    def readTrainLabelData(self):
        print('[LOG] readTrainLabelData start.')
        labels = pd.read_csv(self.trainDataLabelPath)
        self.labels = labels[labels['arrival_date'] < '2017-02-01']
        return labels[labels['arrival_date'] >= '2017-02-01']

    def readTrainData(self):
        print('[LOG] readTrainData start.')
        # read csv file.
        trainData_df = pd.read_csv(self.trainDataPath)

        # refine data, map text to int, drop useless columns, drop na.
        trainData_df = self.refineData(trainData_df)

        # clean data
        ## clean row (should I?)
        trainData_df.drop(trainData_df[trainData_df.adults >= 5].index, inplace=True)
        trainData_df.drop(trainData_df[trainData_df.children >= 10].index, inplace=True)

        # extend reservation
        trainData_df.rename(columns={'arrival_date_year':'year', 'arrival_date_month':'month', 'arrival_date_day_of_month':'day'}, inplace=True)
        trainData_df['arrival_date'] = pd.to_datetime(trainData_df[['year', 'month', 'day']])
        trainData_df['stays_in_nights'] = trainData_df.stays_in_week_nights + trainData_df.stays_in_weekend_nights

        # drop useless columns
        trainData_df.drop(['ID', 'year', 'reservation_status', 'reservation_status_date'], axis=1, inplace=True)
        
        # 
        ## [              |        |                      ]
        ## [ trainData_df | trainY | trainDataReservation ]
        ## [              |        |                      ]
        ## [ --------------------valid------------------- ]
        ## [ validData_df | validY | validDataReservation ]

        # Validation split
        validData_df = trainData_df[trainData_df['arrival_date'] >= '2017-02-01']
        trainData_df = trainData_df[trainData_df['arrival_date'] < '2017-02-01']
        # pop reservation and Y
        trainDataReservation = pd.concat([trainData_df.pop(x) for x in ['arrival_date', 'stays_in_nights']], axis=1)
        validDataReservation = pd.concat([validData_df.pop(x) for x in ['arrival_date', 'stays_in_nights']], axis=1)
        trainY = pd.concat([trainData_df.pop(x) for x in ['is_canceled', 'adr']], axis=1)
        validY = pd.concat([validData_df.pop(x) for x in ['is_canceled', 'adr']], axis=1)


        trainData_df = trainData_df.astype('float64')
        validData_df = validData_df.astype('float64')
        self.X = trainData_df.to_numpy()
        self.Y = trainY.to_numpy()
        self.trainDataReservation = trainDataReservation

        return validData_df.to_numpy(), validY.to_numpy(), validDataReservation

    def getLabelsAcc(self, is_canceled, adr):
        self.trainDataReservation['is_canceled'] = is_canceled
        self.trainDataReservation['adr'] = adr
        acc = 0
        for index, arrival_date in enumerate(self.labels['arrival_date']):
            tmp = self.trainDataReservation.loc[(self.trainDataReservation.is_canceled == 0) & (self.trainDataReservation.arrival_date == arrival_date)]
            # print(tmp)
            if not tmp.empty:
                # tmp['value'] = tmp.stays_in_nights * tmp.adr
                label = self.getLabels((tmp.stays_in_nights * tmp.adr).sum())
            else:  
                label = 0
            acc += (label - self.labels.iloc[index][1])**2
        return sqrt(acc / len(self.labels))

    def oneHot(self):
        self.X = self.oneHotNumpy(self.X)

    def toTensor(self):
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).float()

    def __getitem__(self, index):
        return (self.X[index], self.Y[index][0].long(), self.Y[index][1])

    def __len__(self):
        return len(self.Y)


class TestDataset(DataReader, Dataset):
    """docstring for TestDataset"""
    def __init__(self, dataFolder):
        super(TestDataset, self).__init__()
        self.testDataPath = os.path.join(dataFolder, 'test.csv')
        self.testDataLabelPath = os.path.join(dataFolder, 'test_nolabel.csv')
        self.X, self.testDataReservation = None, None
        self.labels = None

    def readTestData(self):
        print('[LOG] readTestData start.')
        # read csv.
        testData_df = pd.read_csv(self.testDataPath)

        # refine data, map text to int, drop useless columns, drop na.
        testData_df = self.refineData(testData_df)

        # extend reservation
        testData_df.rename(columns={'arrival_date_year':'year', 'arrival_date_month':'month', 'arrival_date_day_of_month':'day'}, inplace=True)
        testData_df['arrival_date'] = pd.to_datetime(testData_df[['year', 'month', 'day']])
        testData_df['stays_in_nights'] = testData_df.stays_in_week_nights + testData_df.stays_in_weekend_nights

        # 
        ## [             |                     ]
        ## [ testData_df | testDataReservation ]
        ## [             |                     ]

        # pop reservation
        testDataReservation = pd.concat([testData_df.pop(x) for x in ['arrival_date', 'stays_in_nights']], axis=1)

        # drop useless columns
        testData_df.drop(['ID', 'year'], axis=1, inplace=True)

        self.X = testData_df.to_numpy()
        self.testDataReservation = testDataReservation

    def readTestLabelData(self):
        print('[LOG] readTestLabelData start.')
        self.labels = pd.read_csv(self.testDataLabelPath)

    def predictLabels(self, is_canceled, adr):
        self.testDataReservation['is_canceled'] = is_canceled
        self.testDataReservation['adr'] = adr
        answers = []
        for index, arrival_date in enumerate(self.labels['arrival_date']):
            tmp = self.testDataReservation.loc[(self.testDataReservation.is_canceled == 0) & (self.testDataReservation.arrival_date == arrival_date)]
            if not tmp.empty:
                label = self.getLabels((tmp.stays_in_nights * tmp.adr).sum())
            else:
                label = 0.0
            answers.append(label)
        self.labels['label'] = answers
        return self.labels

    def oneHot(self):
        self.X = self.oneHotNumpy(self.X)

    def toTensor(self):
        self.X = torch.from_numpy(self.X).float()

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)