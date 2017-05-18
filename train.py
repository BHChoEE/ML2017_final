import pandas as pd
import numpy as np



if __name__ == '__main__':
######### data processing ############

    #import train.csv and cut apart continuous and discrete feature
    train = pd.read_csv('./data/train.csv', header = None)
    train = np.array(train)
    print('Start Data Parsing...')
    '''
    #all of the discrete data and find there col number
    for i, obj in enumerate(train[0]):
        if obj == 'timestamp':
            print('timestamp                    '+str(i))
        if obj == 'product_type':
            print('product_type                 '+str(i))
        if obj == 'sub_area':
            print('sub_area                     '+str(i))
        if obj == 'culture_objects_top_25':
            print('culture_objects_top_25       '+str(i))
        if obj == 'thermal_power_plant_raion':
            print('thermal_power_plant_raion    '+str(i))
        if obj == 'detention_facility_raion':
            print('detention_facility_raion     '+str(i))
        if obj == 'water_1line':
            print('water_1line                  '+str(i))
        if obj == 'big_road1_1line':
            print('big_road1_1line              '+str(i))
        if obj == 'railroad_1line':
            print('railroad_1line               '+str(i))
        if obj == 'ecology':
            print('ecology                      '+str(i))
        if obj == 'price_doc':
            print('price_doc                    '+str(i))
    '''
    print('train.csv has shape of:'+str(train.shape))
    type_list = []
    #cut the feature table down    
    x_train = train[1:30473, 1:292]
    print('x_train has shape of:' + str(x_train.shape))
    print('Start changing discrete data to continuous ones')

    for i, row in enumerate(x_train):
        #print(i)
        #change the yes/no discrete feature to continuous 0 / 1 
        yn_list = [28, 32, 33, 34, 35, 36, 37, 38, 39, 105, 113, 117]
        for j, obj in enumerate(yn_list):
            if x_train[i, obj] == 'yes':
                x_train[i, obj] = 1
            if x_train[i, obj] == 'no':
                x_train[i, obj] = 0
            #print(x_train[i, obj])
        #change ecology(feat.151) to number with 0~4 (0 = no data; 1~4 = degree)
        if x_train[i, 151] == 'poor':
            x_train[i, 151] = 1
        if x_train[i, 151] == 'satisfactory':
            x_train[i, 151] = 2
        if x_train[i, 151] == 'good':
            x_train[i, 151] = 3
        if x_train[i, 151] == 'excellent':
            x_train[i, 151] = 4
        if x_train[i, 151] == 'no data':
            x_train[i, 151] = 0
        #print(x_train[i, 151])
        #change product type to 0/1
        if x_train[i, 10] == 'Investment':
            x_train[i, 10] = 0
        if x_train[1, 10] == 'OwnerOccupier':
            x_train[i, 10] = 1
        #print(x_train[1,10])

        #check how many type of discrete number
        flag = 0
        for j,obj in enumerate(type_list):
            if obj == x_train[i, 11]:
                flag = 1
        if flag == 0:
            type_list.append(x_train[i, 11])

    print('states list number:' +str(len(type_list)))
    x_train_discrete = np.zeros((x_train.shape[0], len(type_list)))
    for i, row in enumerate(x_train):
        for j , comp in enumerate(type_list):
            if comp == x_train[i, 11]:
                x_train[i, 11] = j
        x_train_discrete[i, x_train[i, 11]] = 1
    print('shape of x_train_discrete is:' + str(x_train_discrete.shape))
    y_train = train[1:30473, 291]
    print('shape of y_train is:'+ str(y_train.shape))
    #print(y_train[0])
    #print(x_train[0][290])
    x_train = np.delete(x_train,[11, 290], axis = 1)
    print('shape of x_train is:' + str(x_train.shape))

    '''
    #import macro.csv(data for the economy representation correspond to the day)
    train_date = pd.read_csv('./data/macro.csv')
    train_date = np.array(train_date)
    #print(train_date.shape)
    #print(train_date[2483, 0]) #last date = 2016-16-19        
    '''