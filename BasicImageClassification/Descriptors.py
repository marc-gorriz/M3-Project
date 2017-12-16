import time

import cv2
import numpy as np


class SIFT:
    def __init__(self, nfeatures):
        """

        :param nfeatures:
        """
        self.nfeatures = nfeatures

        # TODO: detect if python2 or python3 is used
        # create the SIFT detector object
        self.SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)  # @python3
        # self.SIFTdetector = cv2.SIFT(nfeatures=self.nfeatures) @python2

    def image_features(self, ima):
        """

        :param ima:
        :return:
        """

        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = self.SIFTdetector.detectAndCompute(gray, None)
        return kpt, des

    def extract_features(self, data_dictionary):
        """

        :param data:
        :return:
        """
        start_time = time.time()

        Train_descriptors = []

        for idx in range(len(data_dictionary['filenames'])):
            ima = cv2.imread(data_dictionary['filenames'][idx])
            kpt, des = self.image_features(ima)
            Train_descriptors.append(des)

        data = Train_descriptors[0]
        labels = np.array([data_dictionary['labels'][0]] * Train_descriptors[0].shape[0])

        for idx in range(1, len(Train_descriptors)):
            data = np.vstack((data, Train_descriptors[idx]))
            labels = np.hstack((labels, np.array([data_dictionary['labels'][idx]] * Train_descriptors[idx].shape[0])))

        print('SIFT features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return data, labels

class SURF:
    def __init__(self, nOctaves=4, nOctaveLayers=2):
        """

        :param nfeatures:
        """
        self.nOctaves=nOctaves
        self.nOctaveLayers=nOctaveLayers


        self.SURFdetector = cv2.xfeatures2d.SURF_create(nOctaves=self.nOctaves, nOctaveLayers=self.nOctaveLayers)

    def image_features(self, ima):
        """

        :param ima:
        :return:
        """

        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = self.SURFdetector.detectAndCompute(gray, None)
        return kpt, des

    def extract_features(self, data_dictionary):
        """

        :param data:
        :return:
        """
        start_time = time.time()

        Train_descriptors = []

        for idx in range(len(data_dictionary['filenames'])):
            ima = cv2.imread(data_dictionary['filenames'][idx])
            kpt, des = self.image_features(ima)
            Train_descriptors.append(des)

        data = Train_descriptors[0]
        labels = np.array([data_dictionary['labels'][0]] * Train_descriptors[0].shape[0])

        for idx in range(1, len(Train_descriptors)):
            data = np.vstack((data, Train_descriptors[idx]))
            labels = np.hstack((labels, np.array([data_dictionary['labels'][idx]] * Train_descriptors[idx].shape[0])))

        print('SURF features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return data, labels
