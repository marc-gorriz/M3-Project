from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import glob
import os
import shutil
from random import shuffle

import numpy as np


def path_list(path, th):
	img_list = glob.glob(path + '/*')
	shuffle(img_list)
	return img_list[0:th], img_list[th:len(img_list)-1]

train_data_dir = '../../Databases/MIT_split/train/'
test_data_dir = '../../Databases/MIT_split/test/'

output_train_data_dir = '../../Databases/MIT_split_validation/train/'
output_validation_data_dir = '../../Databases/MIT_split_validation/validation/'
output_test_data_dir = '../../Databases/MIT_split_validation/test/'

if not os.path.exists(output_train_data_dir):
    os.makedirs(output_train_data_dir)   
if not os.path.exists(output_validation_data_dir):
    os.makedirs(output_validation_data_dir)
if not os.path.exists(output_test_data_dir):
    os.makedirs(output_test_data_dir)

classes = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

#number files per class to create validation set.
nb_train_class = 25
nb_test_class = 40

output_train = []
output_validation = []
output_test = []

for c in classes:
	
	train_path = output_train_data_dir + c + "/"
	validation_path = output_validation_data_dir + c + "/"
	test_path = output_test_data_dir + c + "/"
	
	if not os.path.exists(train_path):
		os.makedirs(train_path)
	if not os.path.exists(validation_path):
		os.makedirs(validation_path)
	if not os.path.exists(test_path):
		os.makedirs(test_path)
	
	train_validation_names, train_names = path_list(train_data_dir + c, nb_train_class)
	test_validation_names, test_names = path_list(test_data_dir + c, nb_test_class)
	
	for name in train_validation_names:
		shutil.copy(name, validation_path)
	for name in train_names:
		shutil.copy(name, train_path)
	for name in test_validation_names:
		shutil.copy(name, validation_path)
	for name in test_names:
		shutil.copy(name, test_path)