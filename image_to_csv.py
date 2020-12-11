import numpy
import cv2 as cv
import os
import csv


def create_file_list(my_dir, format='.jpg'):
    file_list = []
    print(my_dir)
    for root, dirs, files in os.walk(my_dir, topdown=False):
        for name in files:
            if name.endswith(format):
                full_name = os.path.join(root, name)
                file_list.append(full_name)
    return file_list


datadir_train = 'datasets/trainingSet'
datadir_test = 'datasets/test_set'

for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    path = os.path.join(datadir_train, i)
    for file in os.listdir(path):
        img_file = cv.imread(os.path.join(path, file))
        img_gray = cv.cvtColor(img_file, cv.COLOR_BGR2GRAY)
        img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        img_thresh = cv.resize(img_thresh, (32, 32))
        value = numpy.asarray(img_thresh, dtype=numpy.int)
        value = value.flatten()
        value = numpy.insert(value, 0, int(i)+26, axis=0)
        with open("digits_dataset_train.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)

for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    path = os.path.join(datadir_test, i)
    for file in os.listdir(path):
        img_file = cv.imread(os.path.join(path, file))
        img_gray = cv.cvtColor(img_file, cv.COLOR_BGR2GRAY)
        img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        img_thresh = cv.resize(img_thresh, (32, 32))
        value = numpy.asarray(img_thresh, dtype=numpy.int)
        value = value.flatten()
        value = numpy.insert(value, 0, int(i)+26, axis=0)
        with open("digits_dataset_test.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(value)
