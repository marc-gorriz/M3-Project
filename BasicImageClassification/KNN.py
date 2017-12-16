import pickle

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, nneighbors, features_descriptor, njobs=-1):
        """

        :param nneighbors:
        :param njobs:
        """
        self.nneighbors = nneighbors
        self.features_descriptor = features_descriptor
        self.njobs = njobs

    def train(self, data_dictionary):
        """

        :param data:
        :param model_path:
        :param model_name:
        :return:
        """

        data, labels = self.features_descriptor.extract_features(data_dictionary)

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
            _, des = self.features_descriptor.image_features(ima)

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
