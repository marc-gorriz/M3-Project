import os
import random

import pickle
import numpy as np
from sklearn.model_selection import KFold

class Input:
    def __init__(self, workingPath, nsamplesClass, train_method='kfold', k=5, shuffle=True):
        """

        :param workingPath:
        :param nsamplesClass:
        :param train_method:
        :param k:
        :param shuffle:
        """
        self.nsamplesClass = nsamplesClass
        self.train_method = train_method
        self.shuffle = shuffle

        self.train_images_filenames = np.array(pickle.load(open(os.path.join(workingPath, 'train_images_filenames.dat'), 'rb')))
        self.train_labels = np.array(pickle.load(open(os.path.join(workingPath, 'train_labels.dat'), 'rb')))
        self.test_images_filenames = np.array(pickle.load(open(os.path.join(workingPath, 'test_images_filenames.dat'), 'rb')))
        self.test_labels = np.array(pickle.load(open(os.path.join(workingPath, 'test_labels.dat'), 'rb')))

        self.classes = np.unique(self.train_labels)  # use set(self.train_labels)
        self.nTrain = len(self.train_images_filenames)

        self.reduce_data()

        if train_method not in ["fixed", "kfold"]:
            print("Invalid data method")
            #exit()

        if train_method == 'kfold':
            id_train = np.arange(self.nTrain)
            self.kfold_generator = KFold(n_splits=k, shuffle = True).split(id_train)

    def reduce_data(self):
        """

        :return:
        """
        if self.shuffle == True:
            r = random.random()
            random.shuffle(self.train_images_filenames, lambda: r)
            random.shuffle(self.train_labels, lambda: r)

        idClasses = np.array([], dtype=int)
        for c in self.classes:
            idClasses = np.hstack((idClasses, np.where(self.train_labels == c)[0][0:self.nsamplesClass]))


        self.train_images_filenames = self.train_images_filenames[idClasses]
        self.train_labels = self.train_labels[idClasses]

        if self.shuffle == True:
            r = random.random()
            random.shuffle(self.train_images_filenames, lambda: r)
            random.shuffle(self.train_labels, lambda: r)

        #train lenght adjusted to nSamples
        self.nTrain = len(self.train_images_filenames)

    def get_labeled_data(self):
        """

        :return:
        """


        if self.train_method == 'fixed':
            # return 80% train and 20% validation
            train_idx = np.arange(int(np.ceil(self.nTrain * 0.8)))
            validation_idx = np.arange(int(np.ceil(self.nTrain * 0.8)), self.nTrain)

            return {"train_images_filenames": self.train_images_filenames[train_idx],
                    "train_labels": self.train_labels[train_idx],
                    "validation_images_filenames": self.train_images_filenames[validation_idx],
                    "validation_labels": self.train_labels[validation_idx]}

        elif self.train_method == 'kfold':

            train_idx, validation_idx = self.kfold_generator.next()
            return {"train_images_filenames": self.train_images_filenames[train_idx],
                    "train_labels": self.train_labels[train_idx],
                    "validation_images_filenames": self.train_images_filenames[validation_idx],
                    "validation_labels": self.train_labels[validation_idx]}

    def get_test_data(self):
        """

        :return:
        """
        return {"test_images_filenames": self.test_images_filenames,
                "test_labels": self.test_labels}

    def method_data_dictionary(self, data_dictionary, method):
        """

        :param data_dictionary:
        :param method:
        :return:
        """

        if method in ['train', 'validation', 'test']:
            return {"filenames": data_dictionary[str(method) + '_images_filenames'],
                    "labels": data_dictionary[str(method) + '_labels']}
        else:
            print("Invalid method, only train, validation and test are axpected")
