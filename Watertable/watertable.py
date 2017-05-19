#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
"""

import os
import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    """ Main function """
    # Get the data id and label
    label_file = os.path.join(BASE_DIR, 'data/train_labels.csv')
    label_dict = {'functional': 0, 'non functional': 1,
                  'functional needs repair': 2}
    id_train = []
    y_train = []
    with open(label_file, 'r') as lab_file:
        for row in csv.DictReader(lab_file):
            y_train.append(label_dict[row['status_group']])
            id_train.append(int(row['id']))

    y_train = np.array(y_train)
    id_train = np.array(id_train)

    # Get the feature
    # Feature listed in here is useless in my opinion or similar to other
    useless_features = ['id', 'funder', 'installer', 'region', 'recorded_by',
                        'extraction_type_group', 'payment_type', 'quantity_group',
                        'source_type', 'waterpoint_type_group', 'wpt_name', 'subvillage']
    data_file = os.path.join(BASE_DIR, 'data/train_values.csv')
    data = []
    with open(data_file, 'r') as dat_file:
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

    data_size = len(data)

    # Feature listed in here should be normalized
    continuous_features = ['amount_tsh', 'date_recorded', 'gps_height', 'longitude',
                           'latitude', 'num_private', 'population', 'construction_year']

    # date_recorded should be transformed to the days since it has been recorded
    for i in range(data_size):
        year = int(data[i]['date_recorded'][:4])
        month = int(data[i]['date_recorded'][5:7])
        date = int(data[i]['date_recorded'][8:10])
        day = (30 - date) + 30 * (12 - month) + 365 * (2017 - year)
        data[i]['date_recorded'] = day

    # Normalization
    tmp = [[data[i][feature] for feature in continuous_features]
           for i in range(data_size)]
    tmp = np.array(tmp, dtype=float)
    mean = np.mean(tmp, axis=0)
    std = np.std(tmp, axis=0)
    # This array can be then concatenate with one-hot encoded discrete data
    norm_data = (tmp - mean) / std

    # Other Feature is discrete and should be dealed with one-hot encoding
    discrete_features = ['basin', 'region_code', 'district_code', 'lga', 'ward', 'public_meeting',
                         'scheme_management', 'scheme_name', 'permit', 'extraction_type',
                         'extraction_type_class', 'management', 'management_group', 'payment',
                         'water_quality', 'quality_group', 'quantity', 'source', 'source_class',
                         'waterpoint_type']

    for feature in discrete_features:
        # Temp is a list of dictionary. Each dictionary only contains 1 kind of feature
        tmp = [{feature: data[i][feature]} for i in range(data_size)]
        vec = DictVectorizer()
        # Can be concatenate to x_train, not yet concatenate
        data_array = vec.fit_transform(tmp).toarray()
        # Can be concatenate to the feature labels
        data_feature = vec.get_feature_names()
        print('The size of {}: '.format(feature), end='')
        print(data_array.shape)

if __name__ == '__main__':

    main()
