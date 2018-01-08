import cv2
import numpy as np
import pickle
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
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
            # values, counts = np.unique(prediction, return_counts=True)
            # predictions = np.hstack((predictions, values[np.argmax(counts)]))
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
    def __init__(self, kernel='rbf', C=10, gamma=.0001):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma

    def train(self, train_visual_words, train_data):
        print('Training the SVM classifier...')

        init = time.time()

        stdSlr = StandardScaler().fit(train_visual_words)

        # hardcode
        with open("/imatge/mgorriz/work/master/models/session02/test1/stdSlr.pkl", 'wb') as file:
            pickle.dump(stdSlr, file)

        D_scaled = stdSlr.transform(train_visual_words)
        model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma).fit(D_scaled, train_data['labels'])

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

        return model

    def predict(self, model, test_visual_words, test_data):
        init = time.time()

        # hardcode
        with open("/imatge/mgorriz/work/master/models/session02/test1/stdSlr.pkl", 'rb') as file:
            stdSlr = pickle.load(file)

        D_scaled = stdSlr.transform(test_visual_words)
        predictions = model.predict(D_scaled)

        return predictions

    def cross_validation(self, train_visual_words, train_data):
        # TODO: parameters of this function: tuned_parameter and scores, and n_splits, ...
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 2e-3, 1e-4],
                             'C': [1, 10, 100]},
                            {'kernel': ['linear'], 'C': [1, 10, 100]}]

        scores = ['precision', 'recall']

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
            grid = GridSearchCV(svm.SVC(), tuned_parameters, cv=kfold,
                                scoring='%s_macro' % score)
            stdSlr = StandardScaler().fit(train_visual_words)
            D_scaled = stdSlr.transform(train_visual_words)
            grid.fit(D_scaled, train_data['labels'])
            print("Best parameters: %s Accuracy: %0.2f" % (grid.best_params_, grid.best_score_))

            print()
            print("Grid scores on development set:")
            print()
            means = grid.cv_results_['mean_test_score']
            stds = grid.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, grid.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()

            print()

            y_true, y_pred = train_data['labels'], grid.predict(D_scaled)
            print(classification_report(y_true, y_pred))

        # hardcode
        with open("/imatge/mgorriz/work/master/models/session02/test1/best_params_svm.pkl", 'wb') as file:
            pickle.dump(grid.best_params_, file)

        return grid.best_params_

    def save_model(self, model, path):
        with open(path, 'wb') as file:
            pickle.dump(model, file)

    def load_model(self, path):
        with open(path, 'rb') as file:
            model = pickle.load(file)

        return model
