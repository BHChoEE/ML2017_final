import numpy as np 
import pandas as pd
from train import RMSLE
from keras.models import load_model

############### DataPath ################
train_data_path = './data/train.csv'
macro_data_path = './data/macro.csv'
test_data_path = './test/test.csv'
output_path = './test/ans.csv'

############# parameters ################


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


#####################################
######      Main Function      ######           
#####################################

def main():
    ### pre processing data to reasonable figure
    ( _, test_feat, test_discrete, test_time_feat) = read_data(test_data_path, False)
    test_feat_mean = np.nanmean(test_feat, axis = 0)
    test_feat_std = np.nanstd(test_feat, axis = 0)
    for i, obj in enumerate(test_feat):
        for j, col in enumerate(obj):
            if np.isnan(test_feat[i, j]):
                test_feat[i, j] = test_feat_mean[j]
    ### normalize the training data and testing data
    test_feat = (test_feat - test_feat_mean) / test_feat_std

    #X_test = test_feat[0:test_feat.shape[0], 0:2]
    X_test = np.append(test_feat, test_discrete, axis = 1)
    print('The shape of X_test is : ', X_test.shape)
    print(X_test[0:10])
    ### predict test.csv
    model = load_model('train.h5',custom_objects = {'RMSLE': RMSLE})
    Y_test = model.predict(X_test).astype('float')
    #print(Y_test)

    with open (output_path, 'w') as output:
        print('id,price_doc', file = output)
        for i, obj in enumerate(Y_test):
            print(str(i+30474) + ',' + str(Y_test[i,0]), file = output)
   

if __name__ == '__main__':
    main()