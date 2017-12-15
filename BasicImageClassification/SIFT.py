import cv2


class SIFT:
    def __init__(self, nfeatures):
        """

        :param nfeatures:
        """
        self.nfeatures = nfeatures

        # TODO: detect if python2 or python3 is used
        # create the SIFT detector object
        self.SIFTdetector = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures) #@python3
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
        Train_descriptors = []

        for idx in range(len(data_dictionary['filenames'])):
            ima = cv2.imread(data_dictionary['filenames'][idx])
            kpt, des = self.image_features(ima)
            Train_descriptors.append(des)
        return Train_descriptors
