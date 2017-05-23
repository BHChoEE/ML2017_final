#!/usr/bin/env python
# -- coding: utf-8 --
"""
Predict the operating condition of waterpoints
Testing Part
"""

import os
import pickle
from argparse import ArgumentParser
from keras.models import load_model
from train import read_dataset

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='data/values.csv', help='Input file path')
    parser.add_argument('--output', type=str, default='res.csv', help='Output file path')
    parser.add_argument('--model', type=str, default='best', help='Use which model')
    args = parser.parse_args()

    data, id_value = read_dataset(os.path.join(BASE_DIR, args.input), '')
    test_data = data[-14850:]
    id_value = id_value[-14850:]

    # with open('./model/clf.pkl', 'rb') as clf_file:
    #     clf = pickle.load(clf_file)

    # res = clf.predict(test_data)

    model = load_model(os.path.join(MODEL_DIR, "{:s}_model.hdf5".format(args.model)))
    model.summary()

    res = model.predict(test_data, batch_size=128)
    res = res.argmax(axis=1)

    with open(os.path.join(BASE_DIR, args.output), 'w') as output_file:
        print('id,status_group', file=output_file)
        for idx, item in enumerate(id_value):
            if res[idx] == 0:
                status = 'functional'
            elif res[idx] == 1:
                status = 'non functional'
            elif res[idx] == 2:
                status = 'functional needs repair'
            print("{:d},{:s}".format(int(item), status), file=output_file)

if __name__ == '__main__':

    main()
