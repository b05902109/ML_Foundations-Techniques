import pandas as pd
import numpy as np
import argparse, os, pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', default=None, type=str, help='Data folder')
    parser.add_argument('-output', default=None, type=str, help='Data folder')
    arg = parser.parse_args()

    files = (arg.input).split('=')
    files_num = len(files)
    print('[LOG] emsemble start.')
    print('[LOG] file number:', files_num)
    for idx in range(len(files)):
        print('[LOG] file %d: %s'%(idx, files[idx]))

    arrival_date = None
    label = None
    for file in files:
        df = pd.read_csv(file)
        if arrival_date is None:
            arrival_date = df['arrival_date'].values
        
        if label is None:
            label = df['label'].to_numpy()
        else:
            label = np.c_[label, df['label'].to_numpy()]

    ensemble_label = np.array([])
    for row in label:
        tmp = np.bincount(row.astype(int))
        tmp = np.argmax(tmp)
        ensemble_label = np.append(ensemble_label, tmp)

    print(arrival_date.shape, ensemble_label.shape)
    predict = pd.DataFrame()
    predict['arrival_date'] = arrival_date
    predict['label'] = ensemble_label.astype(float)
    predict.to_csv(arg.output, index=False)

# python3 ensemble.py -input predict_XGB_XGB_oneHot.csv=predict_XGB_oneHot_XGB_oneHot.csv=predict_XGB_oneHot_NN_oneHot.csv -output predict_best3.csv