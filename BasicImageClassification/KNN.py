import os
import pickle

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from Descriptors import SIFT, SURF


class KNN:
    def __init__(self, nneighbors, descriptor='SIFT', njobs=-1):
        """

        :param nneighbors:
        :param njobs:
        """
        self.nneighbors = nneighbors
        self.descriptor = descriptor
        self.njobs = njobs

        if self.descriptor == 'SIFT':
            self.mySIFT = SIFT(nfeatures=100)

        elif self.descriptor == 'SURF':
            self.mySURF = SURF()

        else:
            print('Invalid descriptor')

    def train(self, data_dictionary):
        """

        :param data:
        :param model_path:
        :param model_name:
        :return:
        """

        data, labels = self.mySIFT.extract_features(data_dictionary)

        model = KNeighborsClassifier(n_neighbors=self.nneighbors, n_jobs=self.njobs)
        model.fit(data, labels)

        return model

    def predict(self, data, model, display=True):
        """

        :param data:
        :param model_path:
        :param model_name:
        :return:
        """
        nFiles = len(data)
        predictions = np.array([])

        for i in range(len(data)):

            if display == True and i % 50 == 0:
                print("Prediction %d/%d" % (i, nFiles))


            filename = data[i]
            ima = cv2.imread(filename)
            _, des = self.mySIFT.image_features(ima)

            prediction = model.predict(des)
            values, counts = np.unique(prediction, return_counts=True)
            predictions = np.hstack((predictions, values[np.argmax(counts)]))

        return predictions

    def save_model(self, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model
