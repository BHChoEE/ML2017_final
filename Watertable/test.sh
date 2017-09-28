#!/bin/bash
wget 'https://www.dropbox.com/s/011getdx6n8x7zy/model.tgz?dl=1'
tar zxvf model.tgz?dl=1
python3 -B test.py --xgb
rm -rf model model.tgz?dl=1
