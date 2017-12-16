import os
import pickle

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from SIFT import SIFT


class KNN:
    def __init__(self, nneighbors, njobs=-1):
        """

        :param nneighbors:
        :param njobs:
        """
        self.nneighbors = nneighbors
        self.njobs = njobs

    def process_train_data(self, data_dictionary):
        """

        :param data:
        :return:
        """
        # TODO: extend to other descriptors, put outside and pass as a parameter.
        mySIFT = SIFT(nfeatures=100)
        train_features = mySIFT.extract_features(data_dictionary)

        data = train_features[0]
        labels = np.array([data_dictionary['labels'][0]] * train_features[0].shape[0])

        for idx in range(1, len(train_features)):
            data = np.vstack((data, train_features[idx]))
            labels = np.hstack((labels, np.array([data_dictionary['labels'][idx]] * train_features[idx].shape[0])))

        return data, labels

    def train(self, data_dictionary):
        """

        :param data:
        :param model_path:
        :param model_name:
        :return:
        """

        data, labels = self.process_train_data(data_dictionary)

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
        # TODO: same commentary
        mySIFT = SIFT(nfeatures=100)
        nFiles = len(data)
        predictions = np.array([])

        for i in range(len(data)):

            if display == True and i % 50 == 0:
                print("Prediction %d/%d" % i % nFiles)

            filename = data[i]
            ima = cv2.imread(filename)
            _, des = mySIFT.image_features(ima)

            prediction = model.predict(des)
            values, counts = np.unique(prediction, return_counts=True)
            predictions = np.hstack((predictions, values[np.argmax(counts)]))

        return predictions

    def save_model(self, model, path, filename):
        with open(os.path.join(path, filename), 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path, filename):
        with open(os.path.join(path, filename), 'rb') as file:
            model = pickle.load(file)

        return model
