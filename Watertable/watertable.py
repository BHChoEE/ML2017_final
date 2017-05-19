
import os
import argparse
import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
#from keras.preprocessing.text import Tokenizer
#from sklearn.preprocessing import MultiLabelBinarizer



BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

if __name__ == '__main__':
    # get the data id and label
    labelfile = os.path.join(BASE_DIR, "label.csv")
    labeldict = {'functional':0,'non functional':1,'functional needs repair':2}
    id_train = []
    y_train=[]
    with open(labelfile,'r') as f:
        for row in csv.DictReader(f):
            y_train.append(labeldict[row['status_group']])
            id_train.append(int(row['id']))

    y_train = np.array(y_train)
    id_train = np.array(id_train)

    # get the feature
    # Feature listed in here is useless in my opinion or similar to other
    useless_fea = ['id','funder','installer','region','recorded_by','extraction_type_group',
                   'payment_type','quantity_group','source_type','waterpoint_type_group',
                   'wpt_name','subvillage']
    datafile = os.path.join(BASE_DIR, "train.csv")
    data = []
    with open(datafile,'r') as f:
        for i,row in enumerate(csv.reader(f)):
            #feature_type = list(row.keys())
            #break
            if i == 0:
                feature_type = row
                featuredict = {feature_type[k]:k for k in range(len(feature_type))}
                for key in useless_fea:
                    del featuredict[key]
                #print(featuredict)
            else:
                # data is a list with each element is a dictionary
                # feature can be accessed with data[i][feature]
                data.append({key:row[value]
                     for key,value in featuredict.items()})
    datasize = len(data)

    # Feature listed in here should be normalized
    continuous_fea = ['amount_tsh','date_recorded','gps_height','longitude',
                      'latitude','num_private','population','construction_year']

    # date_recorded should be transformed to the days since it has been recorded
    for i in range(datasize):
        year = int(data[i]['date_recorded'][:4])
        month = int(data[i]['date_recorded'][5:7])
        date = int(data[i]['date_recorded'][8:10])
        day = (30-date) + 30*(12-month) + 365*(2017-year)
        data[i]['date_recorded'] = day

    # Normalization
    temparray = [ [ data[i][feature] for feature in continuous_fea ]
                  for i in range(datasize) ]
    temparray = np.array(temparray,dtype=float)
    mean = np.mean(temparray,axis=0)
    std = np.std(temparray,axis=0)
    normData = (temparray - mean) / std # this array can be then concatenate with one-hot encoded discrete data

    # Other Feature is discrete and should be dealed with one-hot encoding
    discrete_fea = ['basin','region_code','district_code','lga','ward','public_meeting',
                    'scheme_management','scheme_name','permit','extraction_type',
                    'extraction_type_class','management','management_group','payment',
                    'water_quality','quality_group','quantity','source','source_class',
                    'waterpoint_type']

    for feature in discrete_fea:

        # temp is a list of dictionary. Each dictionary only contains 1 kind feature.
        temp = [ { feature : data[i][feature] } for i in range(datasize) ]
        vec = DictVectorizer()
        data_array = vec.fit_transform(temp).toarray() # can be concatenate to x_train, not yet concatenate
        data_feature = vec.get_feature_names() # can be concatenate to the featurelabel
        print('The size of {}:  '.format(feature),end='')
        print(data_array.shape)
