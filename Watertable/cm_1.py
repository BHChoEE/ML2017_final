#!/usr/bin/env python
# -- coding: utf-8 --
"""
Get the confusion matrix of trainset and validset
"""

import os
import pickle
from argparse import ArgumentParser
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
import xgboost as xgb
from train import read_dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import itertools

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

np.random.seed(0)

VALIDATION_SPLIT = 0.1
DROPOUT_RATE = 0.3

def plot_confusion_matrix(cm, classes, validsize,
                          title='Confusion matrix',
                          cmap=plt.cm.coolwarm
                          ):

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title + '({}) valid size({})'.format(sys.argv[1][:-3], validsize))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    """ Main function """
    parser = ArgumentParser()
    #parser.add_argument('-i', '--input', type=str, default='data/values.csv', help='Input file path')
    #parser.add_argument('-o', '--output', type=str, default='res.csv', help='Output file path')
    parser.add_argument('-m', '--model', type=str, default='best', help='Use which model')
    parser.add_argument('-r', '--random', action='store_true', help='Use Random Forest model')
    parser.add_argument('-x', '--xgb', action='store_true', help='Use XGBoost model')
    parser.add_argument('-e', '--ensemble', action='store_true', help='Use ensemble XGBoost model')
    args = parser.parse_args()

    data, train_label = read_dataset(os.path.join(BASE_DIR, 'data/values.csv'),
                                     os.path.join(BASE_DIR, 'data/train_labels.csv'))
    train_data = data[:59400]
    print(train_data.shape)
    if not args.random and not args.xgb and not args.ensemble:
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
        with open('./model/rf.pkl', 'rb') as clf_file:
            clf = pickle.load(clf_file)

        train_ans = clf.predict(x_train)
        val_ans = clf.predict(x_val)

    elif args.xgb:
        dxtrain = xgb.DMatrix(x_train)
        dval = xgb.DMatrix(x_val)
        xgb_params = {
            'objective': 'multi:softmax',
            'booster': 'gbtree',
            'eval_metric': 'merror',
            'num_class': 3,
            'learning_rate': .1,
            'max_depth': 14,
            'colsample_bytree': .4,
            'colsample_bylevel': .4
        }
        model = xgb.Booster(dict(xgb_params))
        model.load_model(args.model)
        train_ans = model.predict(dxtrain).reshape(x_train.shape[0], 1)
        val_ans = model.predict(dval).reshape(x_val.shape[0], 1)

    elif args.ensemble:
        dxtrain = xgb.DMatrix(x_train)
        dval = xgb.DMatrix(x_val)

        for i in range(11):
            xgb_params = {
                'objective': 'multi:softmax',
                'booster': 'gbtree',
                'eval_metric': 'merror',
                'num_class': 3,
                'learning_rate': .1,
                'max_depth': 14,
                'colsample_bytree': .4,
                'colsample_bylevel': .4
            }
            model = xgb.Booster(dict(xgb_params))
            model.load_model("./model/xgb/xgb.model_{:d}".format(i))
            if i == 0:
                train_ans = model.predict(dxtrain).reshape(x_train.shape[0], 1)
                val_ans = model.predict(dval).reshape(x_val.shape[0], 1)
            else:
                train_ans = np.append(train_ans, model.predict(dxtrain).reshape(x_train.shape[0], 1), axis=1)
                val_ans = np.append(val_ans, model.predict(dval).reshape(x_val.shape[0], 1), axis=1)

        for idx, arr in enumerate(train_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            train_ans[idx] = tmp.argmax()

        for idx, arr in enumerate(val_ans):
            tmp = np.array([np.where(arr == 0)[0].shape[0], np.where(arr == 1)[0].shape[0], np.where(arr == 2)[0].shape[0]])
            val_ans[idx] = tmp.argmax()

        train_ans = train_ans[:, 0]
        val_ans = val_ans[:, 0]

    else:
        model = load_model(os.path.join(MODEL_DIR, "{:s}.hdf5".format(args.model)))
        model.summary()

        train_ans = model.predict(x_train, batch_size=128)
        val_ans = model.predict(x_val, batch_size=128)

        train_ans = train_ans.argmax(axis=1)
        val_ans = val_ans.argmax(axis=1)
        y_val = y_val.argmax(axis=1)
        y_train = y_train.argmax(axis=1)

    print("Training accuracy: {:f}".format(accuracy_score(y_train, train_ans)))
    print("Validation accuracy: {:f}".format(accuracy_score(y_val, val_ans)))
    conf_mat_val = confusion_matrix(y_val, val_ans)

    plt.figure()
    plot_confusion_matrix(conf_mat_val,
                          classes=['functional', 'non functional','functional needs repair'],
                          validsize = nb_validation_samples,
                          title = 'Validation Set' )
    plt.show()

    conf_mat_train = confusion_matrix(y_train, train_ans)

    plt.figure()
    plot_confusion_matrix(conf_mat_train,
                          classes=['functional', 'non functional','functional needs repair'],
                          validsize = nb_validation_samples,
                          title = 'Train Set' )
    plt.show()

if __name__ == '__main__':

    main()
