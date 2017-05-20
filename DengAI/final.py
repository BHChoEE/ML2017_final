import numpy as np
import os
import sys
import argparse
import sklearn
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path,'data')
weight_path = os.path.join(base_path,'weight')
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

#train_sj_info_l = np.array([ train_label_sj[i][1:3] for i in range(train_label_sj.shape[0]) ])
#train_iq_info_l = np.array([ train_label_iq[i][1:3] for i in range(train_label_iq.shape[0]) ])
train_sj_label = np.array([ train_label_sj[i][3] for i in range(train_label_sj.shape[0]) ]).astype(float)
train_iq_label = np.array([ train_label_iq[i][3] for i in range(train_label_iq.shape[0]) ]).astype(float)

##Build different model for sj and iq###

model_sj = Sequential()
model_sj.add(Dense(21,input_dim = 21,activation = 'relu'))
model_sj.add(Dropout(0.4))
model_sj.add(Dense(64,activation = 'relu'))
model_sj.add(Dropout(0.25))
model_sj.add(Dense(32,activation = 'relu'))
model_sj.add(Dense(1,activation = 'relu'))
model_sj.summary()
model_sj.compile(loss = 'mean_squared_error',optimizer = 'adam',metrics = ['mae'])

filepath_sj=os.path.join(weight_path,"sj_weights_best.hdf5")
checkpoint = ModelCheckpoint(filepath_sj, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='auto',period = 1)
callbacks_list = [checkpoint]

model_sj.fit(x = train_sj_feat,y = train_sj_label,batch_size = 1024,epochs = 1000,validation_split = 0.1,callbacks = callbacks_list)
model_iq = Sequential()
model_iq.add(Dense(21,input_dim = 21,activation = 'relu'))
model_iq.add(Dropout(0.4))
model_iq.add(Dense(32,activation = 'relu'))
model_iq.add(Dropout(0.25))
model_iq.add(Dense(64,activation = 'relu'))
model_iq.add(Dropout(0.25))
model_iq.add(Dense(32,activation = 'relu'))
model_iq.add(Dense(1,activation = 'relu'))
model_iq.summary()
model_iq.compile(loss = 'mean_squared_error',optimizer = 'adam',metrics = ['mae'])

filepath_iq=os.path.join(weight_path,"iq_weights_best.hdf5")
checkpoint = ModelCheckpoint(filepath_iq, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='auto',period = 1)
callbacks_list = [checkpoint]

model_iq.fit(x = train_iq_feat,y = train_iq_label,batch_size = 512,epochs = 1000,validation_split = 0.1, callbacks = callbacks_list)
###load testing data

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

test_data_sj = np.array(test_data_sj)
test_data_sj[test_data_sj == ''] = eps
test_data_iq = np.array(test_data_iq)
test_data_iq[test_data_iq == ''] = eps

test_sj_info = np.array([ test_data_sj[i][0:3] for i in range(test_data_sj.shape[0]) ])
test_iq_info = np.array([ test_data_iq[i][0:3] for i in range(test_data_iq.shape[0]) ])
test_sj_feat = np.array([ np.insert(test_data_sj[i][4:],0,test_data_sj[i][2]) for i in range(test_data_sj.shape[0]) ]).astype('float')
test_iq_feat = np.array([ np.insert(test_data_iq[i][4:],0,test_data_iq[i][2]) for i in range(test_data_iq.shape[0]) ]).astype('float')

test_sj_feat = normalize(test_sj_feat)
test_iq_feat = normalize(test_iq_feat)

ans_sj = np.round(np.array(model_sj.predict(test_sj_feat))).reshape(-1)
ans_iq = np.round(np.array(model_iq.predict(test_iq_feat))).reshape(-1)

with open('ans.csv','w') as f:
    f.write('city,year,weekofyear,total_cases\n')
    for idx, info in enumerate(test_sj_info):
        print(ans_sj[idx])
        f.write('{},{},{},{}\n'.format(info[0],info[1],info[2],int(ans_sj[idx])))
    for idx, info in enumerate(test_iq_info):
        f.write('{},{},{},{}\n'.format(info[0],info[1],info[2],int(ans_iq[idx])))

