import os
import pickle

import cv2
import numpy as np
from SIFT import SIFT
from sklearn.neighbors import KNeighborsClassifier


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
        mySIFT = SIFT(nfeatures=100)
        train_features = mySIFT.extract_features(data_dictionary)

        data = train_features[0]
        labels = np.array([data['labels'][0]] * train_features.shape[0])

        for idx in range(1, len(train_features)):
            data = np.vstack((data, train_features[idx]))
            labels = np.hstack((labels, np.array([data['labels'][idx]] * train_features[0][idx].shape[0])))

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


    def predict(self, data, model):
        """

        :param data:
        :param model_path:
        :param model_name:
        :return:
        """
        mySIFT = SIFT(nfeatures=100)

        predictions = []
        for i in range(len(data)):
            filename = data[i]
            ima = cv2.imread(filename)
            _, des = mySIFT.image_features(ima)
            predictions.append(model.predict(des))

        return predictions

    def save_model(self, model, model_path, model_name):
        pickle.dump(model, open(os.path.join(model_path, model_name), 'wb'))

    def load_model(self, model_path, model_name):
        model = pickle.load(open(os.path.join(model_path, model_name), 'rb'))

        return model

