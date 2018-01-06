import time

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import cluster
import pickle


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

    def extract_features_simple(self, data_dictionary):

        start_time = time.time()

        Train_descriptors = []

        for idx in range(len(data_dictionary['filenames'])):
            ima = cv2.imread(data_dictionary['filenames'][idx])
            kpt, des = self.image_features(ima)
            Train_descriptors.append(des)

        print('SIFT features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return Train_descriptors


class SURF:
    def __init__(self, nOctaves=4, nOctaveLayers=2):
        """

        :param nfeatures:
        """
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers

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

        des = np.reshape(des, [1, len(des)])

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
        labels = np.array([data_dictionary['labels'][0]])

        for idx in range(1, len(Train_descriptors)):
            data = np.vstack((data, Train_descriptors[idx]))
            labels = np.hstack((labels, np.array([data_dictionary['labels'][idx]])))

        print('HOG features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return data, labels


class BOW:
    def __init__(self, k):
        self.k = k

    def compute_codebook(self, Train_descriptors):
        # Transform everything to numpy arrays
        size_descriptors = Train_descriptors[0].shape[1]
        D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
        startingpoint = 0
        for i in range(len(Train_descriptors)):
            D[startingpoint:startingpoint + len(Train_descriptors[i])] = Train_descriptors[i]
            startingpoint += len(Train_descriptors[i])

        print('Computing kmeans with ' + str(self.k) + ' centroids')

        init = time.time()
        codebook = cluster.MiniBatchKMeans(n_clusters=self.k, verbose=False, batch_size=self.k * 20,
                                           compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(D)

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        return codebook

    def extract_features(self, Train_descriptors, codebook, path=None):

        # get train visual word encoding
        print('Getting Train BoVW representation')
        init = time.time()
        visual_words = np.zeros((len(Train_descriptors), self.k), dtype=np.float32)
        for i in xrange(len(Train_descriptors)):
            words = codebook.predict(Train_descriptors[i])
            visual_words[i, :] = np.bincount(words, minlength=self.k)
        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

        return visual_words

    def save(self, file, filename, type):

        if type == 'codebook':
            with open(filename, 'wb') as file_name:
                pickle.dump(file, file_name)

        elif type == 'visualwords':
            np.save(filename, file)
        else:
            print('invalid file')

    def load(self, filename, type):

        if type == 'codebook':
            with open(filename, 'rb') as file_name:
                file = pickle.load(file_name)

        elif type == 'visualwords':
            file = np.load(filename)

        else:
            file = None
            print('Invalid path')

        return file

