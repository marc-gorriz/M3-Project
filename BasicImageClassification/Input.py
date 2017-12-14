import os

class Input(Object):

    def __init__(self, workingPath, trainImagesFilenames, trainLabels, testImagesFilenames, testLabels):

        self.trainImagesFilenames = os.path.join(workingPath,trainImagesFilenames)
        self.trainLabels = os.path.join(workingPath, trainLabels)

        self.testImagesFilenames = os.path.join(workingPath, testImagesFilenames)
        self.testLabels = os.path.join(workingPath, testLabels)

        # check if exists the files
        #if not(os.path.exists(self.trainImagesFilenames))
        #    print("Error: not found {}".format(self.trainImagesFilenames))
        #    exit()

    def load(self):
        self.train_images_filenames = cPickle.load(open(self.trainImagesFilenames, 'rb'))
        self.train_labels = cPickle.load(open(self.trainLabels, 'rb'))
        self.test_images_filenames = cPickle.load(open(self.testImagesFilenames, 'rb'))
        self.test_labels = cPickle.load(open(self.testLabels, 'rb'))

        self.data = {"train_images_filenames": train_images_filenames,
                "test_images_filenames": test_images_filenames,
                "train_labels": train_labels,
                "test_labels": test_labels}

    def getData(self):
        return self.data