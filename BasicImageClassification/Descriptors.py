import time

import cv2
import numpy as np
from skimage.feature import hog


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


class HOG:
    def __init__(self, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2',
                 feature_vector=True):
        """

        :param nfeatures:
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_vector = feature_vector

    def image_features(self, ima):
        """

        :param ima:
        :return:
        """

        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        des = hog(gray, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                  cells_per_block=self.cells_per_block, block_norm=self.block_norm,
                  feature_vector=self.feature_vector)
        kpt = None

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
            labels = np.hstack((labels, np.array([data_dictionary['labels'][idx]])))

        print('HOG features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return data, labels

