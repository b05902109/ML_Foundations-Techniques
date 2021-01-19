from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import argparse, os, pickle
import numpy as np
from math import sqrt
from dataset import TrainDataset, TestDataset

def getTrainScores(gs):
    results = {}
    runs = 0
    for x,y in zip(list(gs.cv_results_['mean_test_score']), gs.cv_results_['params']):
        results[runs] = 'mean:' + str(x) + 'params' + str(y)
        runs += 1
    best = {'best_mean': gs.best_score_, "best_param":gs.best_params_}
    return results, best

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', default='./data', type=str, help='Data folder')
    parser.add_argument('-ckpt', default='./ckpt', type=str, help='Model checkpoint folder')
    parser.add_argument('-img', default='./img', type=str, help='Image folder')
    parser.add_argument('-output', default='./predict.csv', type=str, help='Predict answers')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--is_canceled', action='store_true')
    parser.add_argument('--adr', action='store_true')
    parser.add_argument('--oneHot', action='store_true')

    arg = parser.parse_args()

    modelName = 'RFC_is_canceled'
    features = None
    if arg.oneHot:
        modelName += '_oneHot'

    if arg.train:
        trainDataset = TrainDataset(arg.input)
        validData_df, validY, validDataReservation = trainDataset.readTrainData()
        validLabels = trainDataset.readTrainLabelData()
        validDataset = TrainDataset('')
        validDataset.initial(validData_df, validY, validDataReservation, validLabels)
        features = None
        if not arg.oneHot:
            features = ['hotel','lead_time','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights',
                        'stays_in_week_nights','adults','children','babies','meal','market_segment','distribution_channel','is_repeated_guest',
                        'previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes',
                        'deposit_type','days_in_waiting_list','customer_type','required_car_parking_spaces','total_of_special_requests']
        else:
            features = list(range(77))
            trainDataset.oneHot()
            validDataset.oneHot()

        if arg.is_canceled:
            # randomForestClassifier = RandomForestClassifier()
            # param_test = {
            #     'n_estimators':range(200,1200,200),
            #     'max_depth':range(3,10,2),
            #     'max_depth' : [4,5,6,7,8],
            #     'criterion' :['gini', 'entropy']
            # }
            # # metrics to consider: f1_micro, f1_macro, roc_auc_ovr
            # gsearch1 = GridSearchCV(estimator = randomForestClassifier, param_grid = param_test, scoring = 'f1_micro', n_jobs=1,verbose=10,cv=5)
            # gsearch1.fit(trainDataset.X, trainDataset.Y[:, 0])
            # results, best = getTrainScores(gsearch1)
            # print(results)
            # print(best)

            randomForestClassifier = None
            if not arg.oneHot:
                randomForestClassifier = RandomForestClassifier(n_estimators=800,max_depth=8,criterion='gini')
            else:
                randomForestClassifier = RandomForestClassifier(n_estimators=1000,max_depth=7,criterion='gini')

            print('[LOG] Fitting model...')
            randomForestClassifier.fit(trainDataset.X, trainDataset.Y[:,0])
            print('[LOG] Fitting done!')
            print('-- Model Report --')
            print('RandomForestClassifier train Accuracy: '+str(accuracy_score(randomForestClassifier.predict(trainDataset.X), trainDataset.Y[:,0])))
            print('RandomForestClassifier valid Accuracy: '+str(accuracy_score(randomForestClassifier.predict(validDataset.X), validDataset.Y[:,0])))

            f, ax = plt.subplots(figsize=(10,5))
            plot = sns.barplot(x=features, y=randomForestClassifier.feature_importances_)
            ax.set_title(modelName + ' Feature Importance')
            plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
            plt.savefig(os.path.join(arg.img, modelName))

            # save model
            pickl = {'model': randomForestClassifier}
            if not os.path.isdir(arg.ckpt):
                os.mkdir(arg.ckpt)
            with open(os.path.join(arg.ckpt, modelName + '.pth'), 'wb') as fp:
                pickle.dump(pickl, fp)


    # if arg.test:
    #     testDataset = TestDataset(arg.input)
    #     reservation = testDataset.readTestData()
    #     testDataset.readTestLabelData()
    #     if arg.oneHot:
    #         testDataset.oneHot()

    #     # labels = None
    #     # is_canceled, adr = np.array([]), np.array([])

    #     model = None

    #     # load model
    #     with open(os.path.join(arg.ckpt, modelName + '.pth'), 'rb') as fp:
    #         data = pickle.load(fp)
    #         model = data['model']
    #     is_canceled = model.predict(testDataset.X)