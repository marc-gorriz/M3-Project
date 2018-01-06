import pickle

import cv2
import time
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
            #values, counts = np.unique(prediction, return_counts=True)
            #predictions = np.hstack((predictions, values[np.argmax(counts)]))
            predictions = np.hstack((predictions, prediction))

        return predictions

    def save_model(self, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model


class SVM:

    def __init__(self, kernel='rbf', C=1, gamma=.002):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma


    def train(self, train_visual_words, train_data):

        print('Training the SVM classifier...')

        init = time.time()

        stdSlr = StandardScaler().fit(train_visual_words)

        #hardcode
        with open("/imatge/mgorriz/work/master/models/session02/test1/stdSlr.pkl", 'wb') as file:
            pickle.dump(stdSlr, file)

        D_scaled = stdSlr.transform(train_visual_words)
        model = svm.SVC(kernel='rbf', C=1, gamma=.002).fit(D_scaled, train_data['labels'])

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

        return model

    def predict(self, model, test_visual_words, test_data):

        init = time.time()

        #hardcode
        with open("/imatge/mgorriz/work/master/models/session02/test1/stdSlr.pkl", 'rb') as file:
            stdSlr = pickle.load(file)

        D_scaled = stdSlr.transform(test_visual_words)
        predictions = model.predict(D_scaled)

        return predictions



    def save_model(self, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model


