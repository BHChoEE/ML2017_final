import numpy as np
import os
import sys
import argparse
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
eps = 1e-7
### my normalize ###
def normalize(data):
    mean = []
    for i in range(data.shape[1]):
        col =  data[:,i]
        eps_count = len(col[col == eps ])
        mean.append(sum(data[:,i])/(data.shape[0]-eps_count))
    mean = np.array(mean)
    std = []
    for i in range(data.shape[1]):
        col =  data[:,i]
        eps_count = len(col[col == eps ])
        std.append( np.sqrt(sum((col[col != eps]-mean[i])*(col[col != eps]-mean[i]))/(data.shape[0]-eps_count)) )
    std = np.array(std)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[j,i] != eps:
                data[j,i] = (data[j,i] - mean[i])/std[i]
    return data

### load training data ###
train_data_sj = []
train_data_iq = []
with open(os.path.join(data_path,'dengue_features_train.csv')) as f:
    for i,line in enumerate(f.readlines()):
        if i == 0:
            continue
        if line.strip().split(',')[0] == 'sj':
            train_data_sj.append(line.strip().split(','))
        elif line.strip().split(',')[0] == 'iq':
            train_data_iq.append(line.strip().split(','))
### convert the '' to eps ###
train_data_sj = np.array(train_data_sj)
train_data_sj[train_data_sj == ''] = eps
train_data_iq = np.array(train_data_iq)
train_data_iq[train_data_iq == ''] = eps

train_sj_info = np.array([ train_data_sj[i][1:4] for i in range(train_data_sj.shape[0]) ])
train_iq_info = np.array([ train_data_iq[i][1:4] for i in range(train_data_iq.shape[0]) ])
train_sj_feat = np.array([ np.insert(train_data_sj[i][4:],0,train_data_sj[i][2]) for i in range(train_data_sj.shape[0]) ]).astype(float)
train_iq_feat = np.array([ np.insert(train_data_sj[i][4:],0,train_data_iq[i][2]) for i in range(train_data_iq.shape[0]) ]).astype(float)

train_sj_feat = normalize(train_sj_feat)
train_iq_feat = normalize(train_iq_feat)



###load labels
train_label_sj = []
train_label_iq = []
with open(os.path.join(data_path,'dengue_labels_train.csv')) as f:
    for i,line in enumerate(f.readlines()):
        if i == 0:
            continue
        if line.strip().split(',')[0] == 'sj':
            train_label_sj.append(line.strip().split(','))
        elif line.strip().split(',')[0] == 'iq':
            train_label_iq.append(line.strip().split(','))

train_label_sj = np.array(train_label_sj)
train_label_iq = np.array(train_label_iq)

train_sj_info_l = np.array([ train_label_sj[i][1:3] for i in range(train_label_sj.shape[0]) ])
train_iq_info_l = np.array([ train_label_iq[i][1:3] for i in range(train_label_iq.shape[0]) ])
train_sj_label = np.array([ train_label_sj[i][3] for i in range(train_label_sj.shape[0]) ]).astype(float)
train_iq_label = np.array([ train_label_iq[i][3] for i in range(train_label_iq.shape[0]) ]).astype(float)

model_sj = Sequential()
model_sj.add(Dense(32,input_dim = 21,activation = 'relu'))
model_sj.add(Dropout(0.5))
model_sj.add(Dense(128,activation = 'relu'))
model_sj.add(Dropout(0.25))
model_sj.add(Dense(128,activation = 'relu'))
model_sj.add(Dropout(0.25))
model_sj.add(Dense(64,activation = 'relu'))
model_sj.add(Dropout(0.25))
model_sj.add(Dense(1,activation = 'relu'))
model_sj.summary()

savelist

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model_sj.compile(loss = 'mean_squared_error',optimizer = 'adam',metrics = ['mae'])
model_sj.fit(x = train_sj_feat,y = train_sj_label,batch_size = 100 ,epochs = 500,validation_split = 0.1)




###load testing data
'''
test_data_sj = []
test_data_iq = []
with open(os.path.join(data_path,'dengue_features_test.csv')) as f:
    for i,line in enumerate(f.readlines()):
        if i == 0:
            continue
        if line.strip().split(',')[0] == 'sj':
            test_data_sj.append(line.strip().split(','))
        elif line.strip().split(',')[0] == 'iq':
            test_data_iq.append(line.strip().split(','))

test_data_sj = np.array(test_data_sj)
test_data_iq = np.array(test_data_iq)

test_sj_info = np.array([ test_data_sj[i][1:4] for i in range(test_data_sj.shape[0]) ])
test_iq_info = np.array([ test_data_iq[i][1:4] for i in range(test_data_iq.shape[0]) ])
test_sj_feat = np.array([ np.insert(test_data_sj[i][4:],0,test_data_sj[i][2]) for i in range(test_data_sj.shape[0]) ])
test_iq_feat = np.array([ np.insert(test_data_iq[i][4:],0,test_data_iq[i][2]) for i in range(test_data_iq.shape[0]) ])
'''
