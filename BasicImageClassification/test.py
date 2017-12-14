import cv2
import numpy as np
import cPickle
import time
# import os

from sklearn.neighbors import KNeighborsClassifier

start = time.time()


def getData():
    # read the train and test files

    # agregar el path como parámetro de entrada a la función
    train_images_filenames = cPickle.load(open('train_images_filenames.dat', 'rb'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat', 'rb'))
    train_labels = cPickle.load(open('train_labels.dat', 'rb'))
    test_labels = cPickle.load(open('test_labels.dat', 'rb'))

    print('Loaded ' + str(len(train_images_filenames)) + ' training images filenames with classes ', set(train_labels))
    print('Loaded ' + str(len(test_images_filenames)) + ' testing images filenames with classes ', set(test_labels))

    data = {"train_images_filenames": train_images_filenames,
            "test_images_filenames": test_images_filenames,
            "train_labels": train_labels,
            "test_labels": test_labels}
    return (data)


def SIFT_featureExtraction(nfeatures, data):
    # create the SIFT detector object

    SIFTdetector = cv2.SIFT(nfeatures=nfeatures)



    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays

    Train_descriptors = []
    Train_label_per_descriptor = []

    for i in range(len(data['train_images_filenames'])):
        filename = data["train_images_filenames"][i]
        if Train_label_per_descriptor.count(data["train_labels"][i]) < 30:
            print
            'Reading image ' + filename
            ima = cv2.imread(filename)
            gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
            kpt, des = SIFTdetector.detectAndCompute(gray, None)
            Train_descriptors.append(des)
            Train_label_per_descriptor.append(train_labels[i])
            print
            str(len(kpt)) + ' extracted keypoints and descriptors'

    # Transform everything to numpy arrays

    D = Train_descriptors[0]
    L = np.array([Train_label_per_descriptor[0]] * Train_descriptors[0].shape[0])

    for i in range(1, len(Train_descriptors)):
        D = np.vstack((D, Train_descriptors[i]))
        L = np.hstack((L, np.array([Train_label_per_descriptor[i]] * Train_descriptors[i].shape[0])))

    return (Train_descriptors, Train_label_per_descriptor)





# Train a k-nn classifier

print
'Training the knn classifier...'
myknn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
myknn.fit(D, L)
print
'Done!'

# get all the test data and predict their labels

numtestimages = 0
numcorrect = 0
for i in range(len(test_images_filenames)):
    filename = test_images_filenames[i]
    ima = cv2.imread(filename)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
    kpt, des = SIFTdetector.detectAndCompute(gray, None)
    predictions = myknn.predict(des)
    values, counts = np.unique(predictions, return_counts=True)
    predictedclass = values[np.argmax(counts)]
    print
    'image ' + filename + ' was from class ' + test_labels[i] + ' and was predicted ' + predictedclass
    numtestimages += 1
    if predictedclass == test_labels[i]:
        numcorrect += 1

print
'Final accuracy: ' + str(numcorrect * 100.0 / numtestimages)

end = time.time()
print
'Done in ' + str(end - start) + ' secs.'

## 30.48% in 302 secs.
