import os
import random

import pickle
import numpy as np

class Input:
    def __init__(self, workingPath, nsamplesClass, shuffle=True):
        """

        :param workingPath:
        :param nsamplesClass:
        :param train_method:
        :param k:
        :param shuffle:
        """
        self.nsamplesClass = nsamplesClass
        self.shuffle = shuffle

        self.train_images_filenames = np.array(pickle.load(open(os.path.join(workingPath, 'train_images_filenames.dat'), 'rb')))
        self.train_labels = np.array(pickle.load(open(os.path.join(workingPath, 'train_labels.dat'), 'rb')))
        self.test_images_filenames = np.array(pickle.load(open(os.path.join(workingPath, 'test_images_filenames.dat'), 'rb')))
        self.test_labels = np.array(pickle.load(open(os.path.join(workingPath, 'test_labels.dat'), 'rb')))

        self.classes = np.unique(self.train_labels)  # use set(self.train_labels)
        self.nTrain = len(self.train_images_filenames)

        self.reduce_data()

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

        return {"filenames": self.train_images_filenames,
                "labels": self.train_labels}

    def get_test_data(self):
        """

        :return:
        """
        return {"filenames": self.test_images_filenames,
                "labels": self.test_labels}