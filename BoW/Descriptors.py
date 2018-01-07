import time

import cv2
import numpy as np
from skimage.feature import hog
from sklearn import cluster
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


class SIFT:
    def __init__(self, nfeatures, type='SIFT', dense_sift_step=None):
        """

        :param nfeatures:
        :param type: SIFT or DENSE
        """
        self.nfeatures = nfeatures
        self.type = type
        self.dense_sift_step = dense_sift_step

        # create the SIFT detector object
        if cv2.__version__[0] == '2':
            # OpenCV 2
            self.SIFTdetector = cv2.SIFT(nfeatures=self.nfeatures)
        else:
            # OpenCV 3
            self.SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)

    def image_features(self, ima):
        """

        :param ima:
        :return:
        """

        if self.type == 'SIFT':
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = self.SIFTdetector.detectAndCompute(gray, None)
        else:
            # self.type == 'DENSE'


            sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
            kp1 = list()
            for x in range(0, ima.shape[0], self.dense_sift_step):
                for y in range(0, ima.shape[1], self.dense_sift_step):
                    kp1.append(cv2.KeyPoint(x, y, np.random.randint(self.dense_sift_step, 30)))
            kp1 = np.array(kp1)
            kpt, des = sift.compute(ima, kp1)

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

    # used by SVM classifier
    def extract_features_simple(self, data_dictionary):

        start_time = time.time()

        descriptors_list = []

        for idx in range(len(data_dictionary['filenames'])):
            ima = cv2.imread(data_dictionary['filenames'][idx])
            kpt, des = self.image_features(ima)
            descriptors_list.append(des)

        size_descriptors = descriptors_list[0].shape[1]
        len_D = np.sum([len(p) for p in descriptors_list])

        descriptors = np.zeros((len_D, size_descriptors), dtype=np.uint8)
        descriptors_idx = np.zeros(len_D, dtype=int)

        startingpoint = 0

        idx = np.arange(len_D)

        for i in range(len_D):
            len_Di = len(Train_descriptors[i])
            descriptors[startingpoint:startingpoint + len_Di] = descriptors_list[i]
            descriptors_idx[startingpoint:startingpoint + len_Di] = [idx[i]] * len_Di
            startingpoint += len_Di

        print('SIFT features extracted: done in ' + str(time.time() - start_time) + ' secs')

        return descriptors, descriptors_idx


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

    def compute_codebook(self, descriptors):

        print('Computing kmeans with ' + str(self.k) + ' centroids')
        init = time.time()

        codebook = cluster.MiniBatchKMeans(n_clusters=self.k, verbose=False, batch_size=self.k * 20,
                                           compute_labels=False,
                                           reassignment_ratio=10 ** -4, random_state=42)
        codebook.fit(descriptors)

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')
        return codebook, matrix_idx

    def extract_features(self, descriptors, descriptors_idx, codebook, spatial_pyramid=True):

        # get train visual word encoding
        print('Getting Train BoVW representation')
        init = time.time()

        codebook_predictions = codebook.predict(descriptors)

        if not spatial_pyramid:
            init = time.time()
            visual_words = np.zeros((len(Train_descriptors), self.k), dtype=np.float32)

            for i in range(0, train_idx.max() + 1):
                visual_words[i, :] = np.bincount(codebook_predictions[descriptors_idx == i], minlength=self.k)

        else:
            visual_words = extract_pyramid_visual_words(codebook_predictions, descriptors_idx, self.dense_sift_step)

        end = time.time()
        print('Done in ' + str(end - init) + ' secs.')

        return visual_words

    def extract_pyramid_visual_words(self, codebook_predictions, descriptors_idx):

        assert self.dense_sift_step is not None, "Invalid dense_sift_step"

        # constants
        img_shape = [256, 256]
        pyramid_levels = [[1, 1], [2, 2], [4, 4]]
        keypoints_shape = map(int, [np.ceil(img_shape[0] / self.dense_sift_step), np.ceil(img_shape[1] / self.dense_sift_step)])

        total_features = self.k * np.sum([level[0] * level[1] for level in pyramid_levels])
        visual_words = np.zeros(descriptors_idx.max(), total_features, dtype=np.float64)

        for image_idx in range(0, descriptors_indices.max() + 1):

            image_words_grid = np.reshape(codebook_predictions[descriptors_idx == image_idx], keypoints_shape)

            image_representation = np.zeros(self.k * len(pyramid_levels))
            representation_point = 0

            for pyramid_level in range(0, len(pyramid_levels)):
                step_i = int(np.ceil(keypoints_shape[0] / pyramid_levels[pyramid_level][0]))
                step_j = int(np.ceil(keypoints_shape[1] / pyramid_levels[pyramid_level][1]))

                for i in range(0, keypoints_shape[0], step_i):
                    for j in range(0, keypoints_shape[1], step_j):
                        image_representation[representation_point:representation_point+self.k] = np.array(np.bincount(image_words_grid[i:i + step_i, j:j + step_j].reshape(-1), minlength=self.k))
                        representation_point += self.k

                visual_words[image_idx] = image_representation

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
