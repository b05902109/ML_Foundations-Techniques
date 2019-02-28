from load_data import *
from itertools import *
from sklearn import svm
import numpy as np

dim = 2

users = loadUsers()
#users = users.append(users.loc[0], ignore_index=True)
#users['Country'] = users['Country'].rank().astype(int)
users['Age'].fillna(np.mean(users['Age'].dropna()), inplace=True)
user_R, user_C = users.shape
print('Loaded users')

books = loadBooks()
#books = books.append(books.loc[0], ignore_index=True)
#books['Publisher'].fillna('llll', inplace=True)
#books['Publisher'] = books['Publisher'].rank().astype(int)
book_R, book_C = books.shape
print('Loaded books')

uids = {uid: idx for idx, uid in users['User-ID'].iteritems()}
users = np.c_[users.drop(columns='User-ID').values]

ISBNs = {ISBN: idx for idx, ISBN in books['ISBN'].iteritems()}
books = np.c_[books.drop(columns='ISBN').values]

print('Cut to numpy')

train = loadTrainData()
print('Loaded training data')

#print(books)

#trainX = np.ones((test.shape[0], 2))
#trainY = np.ones((test.shape[0], 1))
trainX = []
trainY = []
for _, row in train.iterrows():
    #print(i)
    user_r = uids.get(row[0], 0)
    book_r = ISBNs.get(row[1], 0)
    tmp = [users[user_r][0], books[book_r][0]]
    trainX.append(tmp)
    trainY.append(row[2])
print('train finish')

test = loadTestData()
#prediction = np.ones((test.shape[0],1))
testX = []

for i, row in test.iterrows():
    user_r = uids.get(row[0], 0)
    book_r = ISBNs.get(row[1], 0)
    tmp = [users[user_r][0], books[book_r][0]]
    testX.append(tmp)
print('test finish')

trainN = len(train)
testN = len(test)
predict_table = np.zeros((testN, 10))
partN = int(trainN / 10)
for i in range(0,5):
    clf = svm.SVC()
    clf.fit(trainX[i*partN:(i+1)*partN], trainY[i*partN:(i+1)*partN])
    predict = clf.predict(testX)
    for j in range(testN):
        predict_table[j][i] = predict[j]
    print(i)
print('svm finish')
np.save('svm_predict_1-5_model', predict_table)
'''
testY = np.zeros((testN,1))
for i in range(testN):
    testY[i] = np.argmax(predict_table[i])

saveSubmission(testY)
'''
