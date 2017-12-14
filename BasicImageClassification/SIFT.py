import cv2
import numpy as np


class SIFT:
    def __init__(self, data, classes, nsamplesClass, nfeatures):
        """

        :param data:
        :param classes:
        :param nsamplesClass:
        :param nfeatures:
        """
        self.data = data
        self.classes = classes
        self.nclasses = len(classes)

        self.nsamplesClass = nsamplesClass
        self.nfeatures = nfeatures

    def extract_features(self):
        Train_descriptors = []
        Train_label_per_descriptor = []

        SIFTdetector = cv2.SIFT(nfeatures=self.nfeatures)

        for c in self.classes:
            id_class = np.where(self.data['labels'] == c)[0:self.nsamplesClass]

            for id in id_class:
                ima = cv2.imread(self.data['filenames'][id])
                gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
                kpt, des = SIFTdetector.detectAndCompute(gray, None)
                Train_descriptors.append(des)
                Train_label_per_descriptor.append(c)

        return (Train_descriptors, Train_label_per_descriptor)
