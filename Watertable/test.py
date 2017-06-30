#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
Testing Part
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
from keras.models import load_model
import xgboost as xgb
from train import read_dataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='data/values.csv', help='Input file path')
    parser.add_argument('--output', type=str, default='res.csv', help='Output file path')
    parser.add_argument('--model', type=str, default='best', help='Use which model')
    parser.add_argument('--random', action='store_true', help='Use Random Forest model')
    parser.add_argument('--xgb', action='store_true', help='Use XGBoost model')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    args = parser.parse_args()

    data, id_value = read_dataset(os.path.join(BASE_DIR, args.input), '')
    test_data = data[-14850:]
    id_value = id_value[-14850:]

    if args.random:
        with open('./model/clf.pkl', 'rb') as clf_file:
            clf = pickle.load(clf_file)

        res = clf.predict(test_data)
    elif args.xgb:
        dxtest = xgb.DMatrix(test_data)
        for i in range(11):
            xgb_params = {
                'objective': 'multi:softmax',
                'booster': 'gbtree',
                'eval_metric': 'merror',
                'num_class': 3,
                'eta': .1,
                'max_depth': 14,
                'colsample_bytree': .4,
                'colsample_bylevel': .4,
            }
            model = xgb.Booster(dict(xgb_params))
            model.load_model("./model/xgb/xgb.model_{:d}".format(i))
            if i == 0:
                res = model.predict(dxtest).reshape(test_data.shape[0], 1)
            else:
                res = np.append(res, model.predict(dxtest).reshape(test_data.shape[0], 1), axis=1)

        for idx, arr in enumerate(res):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            res[idx] = tmp.argmax()
        
        res = res[:, 0]
    elif args.ensemble:
        for i in range(5):
            with open("./model/rf_{:d}.pkl".format(i), 'rb') as clf_file:
                clf = pickle.load(clf_file)

            if i == 0:
                res = clf.predict(test_data).reshape(test_data.shape[0], 1)
            else:
                res = np.append(res, clf.predict(test_data).reshape(test_data.shape[0], 1), axis=1)

        dxtest = xgb.DMatrix(test_data)
        xgb_max_depth = [4, 7, 32, 32, 48]
        for i in range(5):
            xgb_params = {
                'objective': 'multi:softmax',
                'booster': 'gbtree',
                'eval_metric': 'merror',
                'num_class': 3,
                'learning_rate': .1,
                'max_depth': xgb_max_depth[i],
                'colsample_bytree': .5,
                'colsample_bylevel': .5
            }
            model = xgb.Booster(dict(xgb_params))
            model.load_model("./model/xgb.model_{:d}".format(i))
            if i == 0:
                res = model.predict(dxtest).reshape(test_data.shape[0], 1)
            else:
                res = np.append(res, model.predict(dxtest).reshape(test_data.shape[0], 1), axis=1)

        for idx, arr in enumerate(res):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            res[idx] = tmp.argmax()
        
        res = res[:, 0]
    else:
        model = load_model(os.path.join(MODEL_DIR, "{:s}_model.hdf5".format(args.model)))
        model.summary()

        res = model.predict(test_data, batch_size=128)
        res = res.argmax(axis=1)

    with open(os.path.join(BASE_DIR, args.output), 'w') as output_file:
        print('id,status_group', file=output_file)
        for idx, item in enumerate(id_value):
            if res[idx] == 0:
                status = 'functional'
            elif res[idx] == 1:
                status = 'non functional'
            elif res[idx] == 2:
                status = 'functional needs repair'
            print("{:d},{:s}".format(int(item), status), file=output_file)

if __name__ == '__main__':

    main()
