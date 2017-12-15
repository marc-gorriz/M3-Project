import cPickle
import os
import random

import numpy as np


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

        self.classes = np.unique(self.train_labels)
        self.nTrain = len(self.train_images_filenames)

        self.train_images_filenames = cPickle.load(open(os.path.join(workingPath, 'train_images_filenames.dat'), 'rb'))
        self.train_labels = cPickle.load(open(os.path.join(workingPath, 'train_labels.dat'), 'rb'))
        self.test_images_filenames = cPickle.load(open(os.path.join(workingPath, 'test_images_filenames.dat'), 'rb'))
        self.test_labels = cPickle.load(open(os.path.join(workingPath, 'test_labels.dat'), 'rb'))

        self.reduce_data()

        if train_method not in ['fixed', 'kfold']:
            print("Invalid data method")

        if train_method == 'kfold':
            self.k = k
            self.k_idx = 0

    def reduce_data(self):
        """

        :return:
        """
        if self.shuffle == True:
            r = random.random()
            random.shuffle(self.train_images_filenames, lambda: r)
            random.shuffle(self.train_labels, lambda: r)

        idClasses = []
        for c in self.classes:
            idClasses.append(np.where(self.train_labels == c)[0:self.nsamplesClass])

        self.train_images_filenames = self.train_images_filenames[idClasses]
        self.train_labels = self.train_labels[idClasses]

    def get_labeled_data(self):
        """

        :return:
        """

        if self.train_method == 'fixed':
            # return 80% train and 20% validation
            train_idx = np.arange(int(np.ceil(self.nTrain * 0.2)))
            validation_idx = np.arange(int(np.ceil(self.nTrain * 0.2)), self.nTrain)

            return {"train_images_filenames": self.train_images_filenames[train_idx],
                    "train_labels": self.train_labels[[train_idx]],
                    "validation_images_filenames": self.test_images_filenames[validation_idx],
                    "validation_labels": self.test_images_filenames[validation_idx]}

        elif self.train_method == 'kfold':
            if self.k_idx < self.k:
                idx = np.arange(self.nTrain)
                try:
                    idx = np.split(idx, self.k)
                except ValueError:
                    loss = idx[self.nTrain - self.nTrain % self.k:self.nTrain]
                    idx = np.split(idx[0:self.nTrain - self.nTrain % self.k], self.k)
                    idx[len(idx) - 1] = np.hstack((idx[len(idx) - 1], loss))

                validation_idx = idx[self.k - 1]
                train_idx = np.hstack(np.delete(idx, self.k - 1))

                self.k_idx += 1

                return {"train_images_filenames": self.train_images_filenames[train_idx],
                        "train_labels": self.train_labels[[train_idx]],
                        "validation_images_filenames": self.test_images_filenames[validation_idx],
                        "validation_labels": self.test_images_filenames[validation_idx]}
            else:
                self.k_idx = 0
                return False

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
        if data_dictionary == False:
            print("No data available: Kfold is over.")
            return False

        if method in ['train', 'validation', 'test']:
            return {"filenames": data_dictionary[str(method) + '_images_filenames'],
                    "labels": data_dictionary[str(method) + '_labels']}
        else:
            print("Invalid method, only train, validation and test are axpected")

