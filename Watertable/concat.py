#!/usr/bin/env python
# -- coding: utf-8 --
"""
Concat training and testing data
"""

def main():
    """ Main function """
    with open('./data/values.csv', 'w') as concat_file:
        with open('./data/train_values.csv', 'r') as train_file:
            for _, row in enumerate(train_file.readlines()):
                print(row, end='', file=concat_file)

        with open('./data/test_values.csv', 'r') as test_file:
            for idx, row in enumerate(test_file.readlines()):
                if idx == 0:
                    continue
                else:
                    print(row, end='', file=concat_file)

if __name__ == '__main__':

    main()
