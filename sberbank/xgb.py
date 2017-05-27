import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

#from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

################ parameters ##################
split_ratio = 0.1


#################
# Util funciton #
#################


#################
# Main funciton #
#################
def main():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./test/test.csv')
    macro = pd.read_csv('./data/macro.csv')
    id_test = test.id
    train.sample(3)

# Any results you write to the current directory are saved as output.

    train_ , train_val = model_selection.train_test_split(train, test_size = split_ratio)
    y_train = train_["price_doc"]
    y_val = train_val["price_doc"]
    x_train = train_.drop(["id", "timestamp", "price_doc"], axis=1)
    x_val = train_val.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

# cut validation set Checkpoint
    '''
    print(np.asarray(train).shape)
    print(np.asarray(y_train).shape)
    print(np.asarray(y_val).shape)
    print(np.asarray(x_train).shape)
    print(np.asarray(x_val).shape)
    '''
#can't merge train with test because the kernel run for very long time

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values)) 
            x_train[c] = lbl.transform(list(x_train[c].values))
            #x_train.drop(c,axis=1,inplace=True)
    for c in x_val.columns:
        if x_val[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_val[c].values)) 
            x_val[c] = lbl.transform(list(x_val[c].values))
            #x_val.drop(c,axis=1,inplace=True)
    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values)) 
            x_test[c] = lbl.transform(list(x_test[c].values))
            #x_test.drop(c,axis=1,inplace=True) 
    ###################### xgboost parameters ######################
    xgb_params1 = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    xgb_params2 = {
        'eta': 0.04,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    xgb_params3 = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    xgb_params4 = {
        'eta': 0.06,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    xgb_params5 = {
        'eta': 0.06,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    ###################### change type of input to models ######################

    dtrain = xgb.DMatrix(x_train, y_train)
    #test on x_val first
    dval = xgb.DMatrix(x_val)
    #real on test sets
    dtest = xgb.DMatrix(x_test)

    ###################### plotting garbage ######################
    '''
    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
        verbose_eval=50, show_stdv=False)
    cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
    '''

    ###################### Model ######################
    num_boost_rounds = 1000#len(cv_output)
    model1 = xgb.train(dict(xgb_params1, silent=0), dtrain, num_boost_round= num_boost_rounds)
    model2 = xgb.train(dict(xgb_params2, silent=0), dtrain, num_boost_round= num_boost_rounds)
    model3 = xgb.train(dict(xgb_params3, silent=0), dtrain, num_boost_round= num_boost_rounds)
    model4 = xgb.train(dict(xgb_params4, silent=0), dtrain, num_boost_round= num_boost_rounds)
    model5 = xgb.train(dict(xgb_params5, silent=0), dtrain, num_boost_round= num_boost_rounds)
    
    #### plotting stuff
    #fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

    y_val_predict1 = model1.predict(dval)
    y_val_predict2 = model2.predict(dval)
    y_val_predict3 = model3.predict(dval)
    y_val_predict4 = model4.predict(dval)
    y_val_predict5 = model5.predict(dval)
    y_val_predict = np.zeros((y_val_predict1.shape[0], 5))
    for i in range(y_val_predict1.shape[0]):
        y_val_predict[i, 0] = y_val_predict1[i]
        y_val_predict[i, 1] = y_val_predict2[i]
        y_val_predict[i, 2] = y_val_predict3[i]
        y_val_predict[i, 3] = y_val_predict4[i]
        y_val_predict[i, 4] = y_val_predict5[i]
    
    ###################### regression of y_val and y_val_predict ######################
    linear_regr = linear_model.LinearRegression()
    linear_regr.fit(y_val_predict, y_val)
    print(linear_regr.coef_)
    
    y_predict1 = model1.predict(dtest)
    y_predict2 = model2.predict(dtest)
    y_predict3 = model3.predict(dtest)
    y_predict4 = model4.predict(dtest)
    y_predict5 = model5.predict(dtest)
    y_predict = np.zeros((y_predict1.shape[0], ))
    for i in range(y_predict.shape[0]):
        y_predict[i] = linear_regr.coef_[0] * y_predict1[i] + linear_regr.coef_[1] * y_predict2[i] 
        y_predict[i] +=linear_regr.coef_[2] * y_predict3[i] + linear_regr.coef_[3] * y_predict4[i] + linear_regr.coef_[4] * y_predict5[i]
    
    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    output.head()

    output.to_csv('./test/xgbSub.csv', index=False)
    
if __name__ == '__main__':
    main()