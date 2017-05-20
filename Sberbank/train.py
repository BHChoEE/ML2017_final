import sys
import numpy as np
import pandas as pd
## keras function
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

############### DataPath ################
train_data_path = './data/train.csv'
macro_data_path = './data/macro.csv'
test_data_path = './test/test.csv'
output_path = './test/ans.csv'

############### Parameters ##############
split_ratio = 0.2
n = split_ratio * 30471
nb_epoch = 15
batch_size = 128

#########################################
########   Utility Function     #########
#########################################

def read_data(path, training):
    print ('Reading data from ',path)
    data = pd.read_csv(path, index_col = 0)
    data = np.array(data)

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
        print(data[0,288])
        feat = np.zeros((data.shape[0], data.shape[1] -1))
        for i, obj in enumerate(data):
            feat[i] = obj[0:288]
        print('Shape of feature is:', feat.shape)
        with open ('./data/index.csv', 'w') as index:
            for i, obj in enumerate(type_index):
                index.write(str(obj)+'\n')
        return label, feat, discrete_feat, time_feat
    else:
        #print(data[0,287])
        feat = np.zeros((data.shape[0], data.shape[1]))
        for i, obj in enumerate(data):
            feat[i] = obj[0:288]
        print('Shape of feature is:', feat.shape)
        return  np.zeros(data.shape[0]) ,feat, np.zeros((data.shape[0], 146)), time_feat
    
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
    tp = K.sum(K.square(K.log(y_pred + 1) - K.log(y_true + 1)))
    return K.sqrt(tp / n)

#####################################
######      Main Function      ######           
#####################################

def main():
    (label, feat, discrete_feat, time_feat) = read_data(train_data_path, True)
    ( _, test_feat, _, test_time_feat) = read_data(test_data_path, False)
    (time_label, time_feat) = read_macro(macro_data_path)
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(feat, label, split_ratio)

    ### model 
    model = Sequential()
    model.add(Dense(128, activation = 'relu',input_shape = (X_train.shape[1], )))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation = 'softplus'))    
    model.summary()
    rmsprop = RMSprop(lr = 0.002, rho = 0.8, epsilon = 1e-7, decay = 1e-4)
    model.compile(
        loss = 'mean_squared_logarithmic_error',
        optimizer = 'rmsprop',
        metrics = [RMSLE]
    )
    '''
    earlystopping = EarlyStopping(
        monitor = 'val_RMSLE',
        patience = 10,
        verbose = 1,
        mode = 'max'
    )
    checkpoint = ModelCheckpoint(
        filepath = 'best.hdf5',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
        monitor = 'val_RMSLE',
        mode = 'max'
    )'''
    hist = model.fit(
        X_train, Y_train,
        #validation = (X_val, Y_val),
        epochs = nb_epoch,
        batch_size = batch_size,
        #allbacks = [earlystopping, checkpoint]
    )
    model.save('train.h5')

    ### predict test.csv
    test_pred = model.predict(test_feat)
    with open (output_path, 'w') as output:
        print('id,price_doc', file = output)
        for i, obj in enumerate(test_pred):
            pr = float(obj[0])
            print(str(i+30474) + ',' + str(pr), file = output)
if __name__ == '__main__':
    main()
