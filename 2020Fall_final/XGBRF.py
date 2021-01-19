from xgboost.sklearn import XGBRFClassifier, XGBRFRegressor
import xgboost as xgb

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

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

    modelName = 'XGBRF'
    if arg.is_canceled:
        modelName += '_is_canceled'
    elif arg.adr:
        modelName += '_adr'
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
            trainDataset.oneHot()
            validDataset.oneHot()
            features = list(range(77))

        if arg.is_canceled:
            # xgbrf_classifier = XGBRFClassifier(
            #         learning_rate=0.1,
            #         n_estimators=1000
            #     )
            # param_test = {
            #     'max_depth':range(3,10,2),
            #     'min_child_weight':range(1,6,2)
            # }
            # #metrics to consider: f1_micro, f1_macro, roc_auc_ovr
            # gsearch1 = GridSearchCV(estimator = xgbrf_classifier, param_grid = param_test, scoring='f1_micro',n_jobs=1,verbose = 10, cv=5)
            # gsearch1.fit(trainDataset.X, trainDataset.Y[:, 0])
            # results, best = getTrainScores(gsearch1)
            # print(results)
            # print(best)

            xgbrf_classifier = None
            if not arg.oneHot:
                xgbrf_classifier = XGBRFClassifier(
                                    learning_rate=0.1,
                                    n_estimators=1000,
                                    max_depth=7,
                                    min_child_weight=5
                                )
            else:
                xgbrf_classifier = XGBRFClassifier(
                                    learning_rate=0.1,
                                    n_estimators=1000,
                                    max_depth=7,
                                    min_child_weight=3
                                )

            print('[LOG] Fitting model...')
            xgbrf_classifier.fit(trainDataset.X, trainDataset.Y[:,0])
            print('[LOG] Fitting done!')
            print('-- Model Report --')
            print('XGBoost train Accuracy: '+str(accuracy_score(xgbrf_classifier.predict(trainDataset.X), trainDataset.Y[:,0])))
            print('XGBoost valid Accuracy: '+str(accuracy_score(xgbrf_classifier.predict(validDataset.X), validDataset.Y[:,0])))

            if not os.path.isdir(arg.img):
                os.mkdir(arg.img)
            f, ax = plt.subplots(figsize=(10,5))
            plot = sns.barplot(x=features, y=xgbrf_classifier.feature_importances_)
            ax.set_title('Feature Importance')
            plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
            plt.savefig(os.path.join(arg.img, modelName))

            # save model
            pickl = {'model': xgbrf_classifier}
            if not os.path.isdir(arg.ckpt):
                os.mkdir(arg.ckpt)
            with open(os.path.join(arg.ckpt, modelName + '.pth'), 'wb') as fp:
                pickle.dump(pickl, fp)

        if arg.adr:
            # xgbrf_regressor = XGBRFRegressor(
            #         learning_rate=0.1,
            #         n_estimators=1000,
            #         objective='reg:squarederror',
            #     )
            # param_test = {
            #     'max_depth':range(3,10,2),
            #     'min_child_weight':range(1,6,2)
            # }
            # #metrics to consider: f1_micro, f1_macro, roc_auc_ovr
            # gsearch1 = GridSearchCV(estimator = xgbrf_regressor, param_grid = param_test, scoring='neg_mean_squared_error',n_jobs=1,verbose = 10, cv=5)
            # gsearch1.fit(trainDataset.X, trainDataset.Y[:, 1])
            # results, best = getTrainScores(gsearch1)
            # print(results)
            # print(best)

            xgbrf_regressor = None
            if not arg.oneHot:
                xgbrf_regressor = XGBRFRegressor(
                                    learning_rate=0.1,
                                    n_estimators=1000,
                                    max_depth=9,
                                    min_child_weight=1,
                                    objective='reg:squarederror',
                                )
            else:
                xgbrf_regressor = XGBRFRegressor(
                                    learning_rate=0.1,
                                    n_estimators=1000,
                                    max_depth=9,
                                    min_child_weight=5,
                                    objective='reg:squarederror',
                                )

            print('[LOG] Fitting model...')
            xgbrf_regressor.fit(trainDataset.X, trainDataset.Y[:,1])
            print('[LOG] Fitting done!')
            print('-- Model Report --')
            print('XGBoost train MSE: '+str(mean_squared_error(xgbrf_regressor.predict(trainDataset.X), trainDataset.Y[:,1])))
            print('XGBoost valid MSE: '+str(mean_squared_error(xgbrf_regressor.predict(validDataset.X), validDataset.Y[:,1])))

            if not os.path.isdir(arg.img):
                os.mkdir(arg.img)
            f, ax = plt.subplots(figsize=(10,5))
            plot = sns.barplot(x=features, y=xgbrf_regressor.feature_importances_)
            ax.set_title('Feature Importance')
            plot.set_xticklabels(plot.get_xticklabels(),rotation='vertical')
            plt.savefig(os.path.join(arg.img, modelName))

            # save model
            pickl = {'model': xgbrf_regressor}
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

# [0]     val-mlogloss:0.66011    val-merror:0.23450      train-mlogloss:0.65899  train-merror:0.23071
# [100]   val-mlogloss:0.42959    val-merror:0.20378      train-mlogloss:0.37136  train-merror:0.16246
# [200]   val-mlogloss:0.42612    val-merror:0.20313      train-mlogloss:0.36275  train-merror:0.16183
# [299]   val-mlogloss:0.42507    val-merror:0.20345      train-mlogloss:0.36032  train-merror:0.16148
# -- Model Report --
# XGBoost Accuracy: 0.7965453154039576
# XGBoost F1-Score (Micro): 0.7965453154039577