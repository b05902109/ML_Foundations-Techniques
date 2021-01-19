import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from xgboost.sklearn import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import argparse, os, pickle
import numpy as np
import pandas as pd
from math import sqrt
from dataset import TestDataset
from model import Net

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', default='./data', type=str, help='Data folder')
    parser.add_argument('-ckpt', default='./ckpt', type=str, help='Model checkpoint folder')
    parser.add_argument('-output', default='./adr.csv', type=str, help='Predict adr')
    parser.add_argument('--adr', default='XGB', type=str, help='adr method')

    arg = parser.parse_args()

    adr_method = (arg.adr).split('_')
    adr_modelName = adr_method[0] + '_adr'

    testDataset_adr = TestDataset(arg.input)
    testDataset_adr.readTestData()
    testDataset_adr.readTestLabelData()

    if 'oneHot' in adr_method:
        adr_modelName += '_oneHot'
        testDataset_adr.oneHot()

    labels = None

    # adr
    adr_predict = np.array([])

    if 'NN' in adr_method:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('[LOG] device: ', device)
        testDataset_adr.toTensor()
        testloader = torch.utils.data.DataLoader(testDataset_adr, batch_size=100, shuffle=False, num_workers=2)
        net = Net(adr=True).to(device)
        state = torch.load(os.path.join(arg.ckpt, adr_modelName + '_batch32_Adam1e-3.pth'),map_location=device)
        net.load_state_dict(state['net'])  
        net.eval()
        with torch.no_grad():
            for batch_idx, (X) in enumerate(testloader):
                X = X.to(device)
                Y2_hat = net(X)

                Y2_hat = Y2_hat.cpu().numpy()
                adr_predict = np.append(adr_predict, Y2_hat)
    elif 'RFC' in adr_method:
        with open(os.path.join(arg.ckpt, adr_modelName + '.pth'), 'rb') as fp:
            data = pickle.load(fp)
            model = data['model']
        adr_predict = model.predict(testDataset_adr.X)
    elif 'XGB' in adr_method:
        with open(os.path.join(arg.ckpt, adr_modelName + '.pth'), 'rb') as fp:
            data = pickle.load(fp)
            model = data['model']
        adr_predict = model.predict(testDataset_adr.X)
    elif 'XGBRF' in adr_method:
        with open(os.path.join(arg.ckpt, adr_modelName + '.pth'), 'rb') as fp:
            data = pickle.load(fp)
            model = data['model']
        adr_predict = model.predict(testDataset_adr.X)
    else:
        print('[ERROR] adr method unknown.')
    # print(adr_predict)
    # print(len(adr_predict))

    df = pd.DataFrame()
    df['ID'] = list(range(91531, 119389+1))
    df['adr'] = adr_predict

    df.to_csv(arg.output, index=False)

# predict_XGB_oneHot_XGB.csv          0.421053
# predict_XGB_oneHot_NN_oneHot.csv    0.381579
# predict_RFC_NN_oneHot.csv           0.710526