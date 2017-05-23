#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
Training Part
"""

import os
import csv
import pickle
from argparse import ArgumentParser
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.3

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

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
    useless_features = ['id', 'funder', 'recorded_by', 'extraction_type_group',
                        'payment', 'quantity_group', 'source_type',
                        'waterpoint_type_group', 'wpt_name', 'subvillage']
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
            data[i]['gps_height'] = 665
        if float(data[i]['longitude']) == 0:
            data[i]['longitude'] = 34.07
        if float(data[i]['latitude']) > -0.1:
            data[i]['latitude'] = -5.7
        if float(data[i]['construction_year']) == 0:
            data[i]['construction_year'] = 1960

    # Normalization
    tmp = [[data[i][feature] for feature in continuous_features]
           for i in range(data_size)]
    tmp = np.array(tmp, dtype=float)
    mean = np.mean(tmp, axis=0)
    std = np.std(tmp, axis=0)
    # This array can be then concatenate with one-hot encoded discrete data
    norm_data = (tmp - mean) / std
    value = norm_data

    # Other Feature is discrete and should be dealed with one-hot encoding
    discrete_features = ['basin', 'lga', 'public_meeting', 'scheme_management', 'permit',
                         'extraction_type', 'region', 'extraction_type_class', 'management',
                         'management_group', 'payment_type', 'water_quality', 'quantity',
                         'source', 'source_class', 'waterpoint_type']

    for feature in discrete_features:
        # Temp is a list of dictionary. Each dictionary only contains 1 kind of feature
        tmp = [{feature: data[i][feature]} for i in range(data_size)]
        vec = DictVectorizer()
        # Can be concatenate to value, not yet concatenate
        data_array = vec.fit_transform(tmp).toarray()
        value = np.append(value, data_array.argmax(axis=1).reshape((data_array.shape[0], 1)), axis=1)
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
    parser.add_argument('--random', action='store_true', help='Use Random Forest model')
    args = parser.parse_args()

    data, train_label = read_dataset(os.path.join(BASE_DIR, 'data/values.csv'),
                                     os.path.join(BASE_DIR, 'data/train_labels.csv'))
    train_data = data[:59400]
    print(train_data.shape)
    if not args.random:
        train_label = to_categorical(train_label)
    indices = np.random.permutation(train_data.shape[0])
    train_data = train_data[indices]
    train_label = train_label[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

    x_train = train_data[:-nb_validation_samples]
    y_train = train_label[:-nb_validation_samples]
    x_val = train_data[-nb_validation_samples:]
    y_val = train_label[-nb_validation_samples:]

    if args.random:
        clf = RandomForestClassifier(min_samples_split=8,
                                     n_estimators=1000,
                                     oob_score=True,
                                     n_jobs=-1)
        clf.fit(x_train, y_train)
        train_ans = clf.predict(x_train)
        val_ans = clf.predict(x_val)
        with open('./model/clf.pkl', 'wb') as clf_file:
            pickle.dump(clf, clf_file)

        print("Training accuracy: {:f}".format(accuracy_score(y_train, train_ans)))
        print("Validation accuracy: {:f}".format(accuracy_score(y_val, val_ans)))
    else:
        model = Sequential()
        model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(DROPOUT_RATE))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        model_name = os.path.join(MODEL_DIR, "{epoch:02d}_model.hdf5")
        checkpoint = ModelCheckpoint(model_name, monitor='val_acc', verbose=0,
                                     save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit(x_train, y_train,
                  epochs=300,
                  batch_size=128,
                  validation_data=(x_val, y_val),
                  callbacks=callbacks_list)

if __name__ == '__main__':

    main()
