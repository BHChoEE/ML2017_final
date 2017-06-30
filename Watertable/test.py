#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
Testing Part
"""

import os
import csv
import pickle
from argparse import ArgumentParser
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from keras.models import load_model
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def read_dataset(value_path, label_path):
    """ Read and process dataset """
    # Get label
    if label_path:
        label_dict = {'functional': 0, 'non functional': 1,
                      'functional needs repair': 2}
        label = []
        with open(label_path, 'r') as lab_file:
            for row in csv.DictReader(lab_file):
                label.append(label_dict[row['status_group']])

        label = np.array(label)

    # Get the feature
    # Feature listed in here is useless in my opinion or similar to other
    useless_features = ['id', 'installer', 'wpt_name', 'num_private', 'subvillage',
                        'region', 'region_code', 'district_code', 'ward', 'recorded_by',
                        'scheme_name', 'extraction_type', 'extraction_type_group',
                        'management_group', 'payment_type', 'water_quality', 'quantity_group',
                        'source_type', 'waterpoint_type_group']
    data = []
    id_value = []
    with open(value_path, 'r') as dat_file:
        for i, row in enumerate(csv.reader(dat_file)):
            if i == 0:
                feature_type = row
                feature_dict = {
                    feature_type[k]: k for k in range(len(feature_type))}
                for key in useless_features:
                    del feature_dict[key]
            else:
                # Data is a list with each element is a dictionary
                # Feature can be accessed with data[i][feature]
                data.append({key: row[value]
                             for key, value in feature_dict.items()})
                id_value.append(row[0])

    data_size = len(data)

    # Feature listed in here should be normalized
    continuous_features = ['amount_tsh', 'date_recorded', 'gps_height', 'longitude',
                           'latitude', 'population', 'construction_year']

    # date_recorded should be transformed to the days since it has been recorded
    # Fill some missing data with average value
    for i in range(data_size):
        year = int(data[i]['date_recorded'][:4])
        month = int(data[i]['date_recorded'][5:7])
        date = int(data[i]['date_recorded'][8:10])
        day = (30 - date) + 30 * (12 - month) + 365 * (2017 - year)
        data[i]['date_recorded'] = day
        if float(data[i]['gps_height']) == 0:
            data[i]['gps_height'] = 1016.97
        if float(data[i]['longitude']) == 0:
            data[i]['longitude'] = 35.15
        if float(data[i]['latitude']) > -0.1:
            data[i]['latitude'] = -5.88
        if float(data[i]['construction_year']) == 0:
            data[i]['construction_year'] = 1997
        if float(data[i]['construction_year']) >= 1960:
            data[i]['construction_year'] = float(data[i]['construction_year']) - 1960

    # Normalization
    tmp = [[data[i][feature] for feature in continuous_features]
           for i in range(data_size)]
    tmp = np.array(tmp, dtype=float)
    mean = np.mean(tmp, axis=0)
    std = np.std(tmp, axis=0)
    # This array can be then concatenate with one-hot encoded discrete data
    norm_data = (tmp - mean) / std
    value = norm_data
    # value = tmp

    # Other Feature is discrete and should be dealed with one-hot encoding
    discrete_features = ['funder', 'basin', 'lga', 'public_meeting', 'scheme_management',
                         'permit', 'extraction_type_class', 'management', 'payment',
                         'quality_group', 'quantity', 'source', 'source_class',
                         'waterpoint_type']

    for feature in discrete_features:
        # Temp is a list of dictionary. Each dictionary only contains 1 kind of feature
        tmp = [{feature: data[i][feature]} for i in range(data_size)]
        vec = DictVectorizer()
        # Can be concatenate to value, not yet concatenate
        data_array = vec.fit_transform(tmp).toarray()
        # Single integer to represent discrete features
        value = np.append(value, data_array.argmax(axis=1).reshape((data_array.shape[0], 1)), axis=1)
        # One hot vector to represent discrete features
        # value = np.append(value, data_array, axis=1)
        # Can be concatenate to the feature labels
        data_feature = vec.get_feature_names()
        # print('The size of {}: '.format(feature), end='')
        # print(data_array.shape)

    if label_path:
        return value, label
    else:
        return value, np.array(id_value)

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
