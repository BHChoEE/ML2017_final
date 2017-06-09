import sys
import math
import numpy as np
import pandas as pd
## keras function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adamax ,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

############### DataPath ################
train_data_path = './data/train.csv'
macro_data_path = './data/macro.csv'
test_data_path = './test/test.csv'
output_path = './test/ans.csv'

############### Parameters ##############
split_ratio = 0.1
n = split_ratio * 30471
nb_epoch = 150
batch_size = 128

#########################################
########   Utility Function     #########
#########################################

def read_data(path, training):
    print ('Reading data from ',path)
    data = pd.read_csv(path, index_col = 0)
    data = np.array(data)
    #print(data[0, 4])
    dis = []
    for i, obj in enumerate(data):
        dis.append(obj[11])
    #print(dis)
    print('Start Converting Data')
    type_index = []
    for i, row in enumerate(data):
        #change the yes/no discrete feature to continuous 0 / 1 
        yn_list = [28, 32, 33, 34, 35, 36, 37, 38, 39, 105, 113, 117]
        for j, obj in enumerate(yn_list):
            if data[i, obj] == 'yes':
                data[i, obj] = 1
            elif data[i, obj] == 'no':
                data[i, obj] = 0
        #change ecology(feat.151) to number with 0~4 (0 = no data; 1~4 = degree)
        if data[i, 151] == 'poor':
            data[i, 151] = 1
        elif data[i, 151] == 'satisfactory':
            data[i, 151] = 2
        elif data[i, 151] == 'good':
            data[i, 151] = 3
        elif data[i, 151] == 'excellent':
            data[i, 151] = 4
        elif data[i, 151] == 'no data':
            data[i, 151] = 0
        #change product type(feat.19) 'Investmen'/'OwnerOccupier' to 0/1
        if data[i, 10] == 'Investment':
            data[i, 10] = 0
        elif data[i, 10] == 'OwnerOccupier':
            data[i, 10] = 1
        #check how many type of discrete number
        flag = 0
        for j,obj in enumerate(type_index):
            if obj == data[i, 11]:
                flag = 1
        if flag == 0:
            type_index.append(data[i, 11])
       
    discrete_feat = np.zeros((data.shape[0], len(type_index)))
    #time_feat = np.zeros(data.shape[0], dtype = str)
    time_feat = []
    for i, row in enumerate(data):
        for j , comp in enumerate(type_index):
            if comp == data[i, 11]:
                data[i, 11] = j
        discrete_feat[i, data[i, 11]] = 1
        time_feat.append(data[i, 0])
    data = np.delete(data, [0,11], axis = 1)
    print('Shape of discrete feature is: ', discrete_feat.shape)
    print('Shape of time feature is: ', len(time_feat))
    
    if training:
        label = np.zeros(data.shape[0])
        for i, obj in enumerate(data):
            label[i] = obj[288]
        #print(data[0,288])
        feat = np.zeros((data.shape[0], data.shape[1] -1))
        for i, obj in enumerate(data):
            feat[i] = obj[0:288]
        print('Shape of feature is:', feat.shape)
        with open ('./data/index.csv', 'w') as index:
            for i, obj in enumerate(type_index):
                index.write(str(obj)+'\n')
        return label, feat, discrete_feat, time_feat
    else:
        #print(data[0,11])
        feat = np.zeros((data.shape[0], data.shape[1]))
        for i, obj in enumerate(data):
            feat[i] = obj[0:288]
        print('Shape of feature is:', feat.shape)
        dis_index = []
        with open('./data/index.csv', 'r') as index:
            for line in index:
                end = line.find('\n')
                dis_index.append(line[0:end])
        #print(dis)
        discrete_feat = np.zeros((feat.shape[0], len(dis_index)))
        for i, obj in enumerate(dis):
            for j, comp in enumerate(dis_index):
                if comp == obj:
                    discrete_feat[i,j] = 1
        #print(discrete_feat.shape)
        return  np.zeros(data.shape[0]) ,feat, discrete_feat, time_feat
    
def read_macro(path ):
    print('Start processing macro.csv')
    macro = pd.read_csv(path, index_col = None)
    macro = np.array(macro)
    label = []
    feat = np.zeros((macro.shape[0], macro.shape[1] - 1), dtype = float)
    for i in range(macro.shape[0]):
        label.append(macro[i, 0])
    print('The Shape of macro label is:', len(label))
    feat = macro[0:macro.shape[0], 1:macro.shape[1]]
    print('The Shape of macro feature is:', feat.shape)
    return label, feat

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

#####################################
#########   custom metrices  ########
#####################################
def RMSLE(y_true,y_pred):
    tp = K.sum(K.square(K.log(y_pred + 1) - K.log(y_true + 1) + K.epsilon()), axis = -1)
    return K.sqrt((tp / n) + K.epsilon())

#####################################
######      Main Function      ######           
#####################################

def main():
    ### pre processing data to reasonable figure
    (label, feat, discrete_feat, time_feat) = read_data(train_data_path, True)
    ( _, test_feat, test_discrete, test_time_feat) = read_data(test_data_path, False)
    (time_label, time_feature) = read_macro(macro_data_path)

    ### to check the validity of feat 
    feat_mean = np.nanmean(feat, axis = 0)
    feat_std = np.nanstd(feat, axis = 0)
    for i, obj in enumerate(feat):
        for j, col in enumerate(obj):
            if np.isnan(feat[i, j]):
                feat[i, j] = feat_mean[j]
    test_feat_mean = np.nanmean(test_feat, axis = 0)
    test_feat_std = np.nanstd(test_feat, axis = 0)
    for i, obj in enumerate(test_feat):
        for j, col in enumerate(obj):
            if np.isnan(test_feat[i, j]):
                test_feat[i, j] = test_feat_mean[j]

    ### normalize the training data and testing data
    feat = (feat - feat_mean) / feat_std
    test_feat = (test_feat - test_feat_mean) / test_feat_std

    ### normailzation of training data
    X_train = np.append(feat, discrete_feat, axis =1)
    print('The shape of X_train is : ', X_train.shape)
    X_test = np.append(test_feat, test_discrete, axis = 1)
    print('The shape of X_test is : ', X_test.shape)
    Y_train = label.reshape((label.shape[0], 1))
    print('The shape of Y_train is :', Y_train.shape)

    ### model 
    model = Sequential()
    model.add(Dense(512, activation = 'relu',input_shape = (X_train.shape[1], )))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(Y_train.shape[1], activation = 'relu'))    
    model.summary()
    rmsprop = RMSprop(lr = 0.002, rho = 0.8, epsilon = 1e-7, decay = 1e-4)
    model.compile(
        loss = 'mean_squared_error',
        optimizer = 'Adamax',
        metrics = [RMSLE]
    )
    
    earlystopping = EarlyStopping(
        monitor = 'val_RMSLE',
        patience = 10,
        verbose = 1,
        mode = 'min'
    )
    checkpoint = ModelCheckpoint(
        filepath = 'best.hdf5',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
        monitor = 'val_RMSLE',
        mode = 'min'
    )
    hist = model.fit(
        X_train, Y_train,
        #validation = (X_val, Y_val),
        validation_split = split_ratio,
        epochs = nb_epoch,
        batch_size = batch_size,
        callbacks = [earlystopping, checkpoint]
    )
    model.save('train.h5')
    
    ### predict test.csv
    Y_test = model.predict(X_test)
    test_pred = Y_test.reshape((Y_test.shape[0], ))
    with open (output_path, 'w') as output:
        print('id,price_doc', file = output)
        for i, obj in enumerate(Y_test):
            pr = float(Y_test[i])
            print(str(i+30474) + ',' + str(pr), file = output)
    
    
if __name__ == '__main__':
    main()
