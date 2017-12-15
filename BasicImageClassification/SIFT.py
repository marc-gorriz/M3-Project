import cv2


class SIFT:
    def __init__(self, nfeatures):
        """

        :param nfeatures:
        """
        self.nfeatures = nfeatures

    def image_features(self, ima):
        """

        :param ima:
        :return:
        """
        SIFTdetector = cv2.SIFT(nfeatures=self.nfeatures)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = SIFTdetector.detectAndCompute(gray, None)
        return kpt, des

    def extract_features(self, data):
        """

        :param data:
        :return:
        """
        Train_descriptors = []

        for idx in range(len(data['filenames'])):
            ima = cv2.imread(data['filenames'][idx])
            kpt, des = self.image_features(ima)
            Train_descriptors.append(des)
            print(str(len(kpt)) + ' extracted keypoints and descriptors')
        return Train_descriptors
