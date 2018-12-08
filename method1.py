from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

# ------------------------------------Pre-processing--------------------------------
print("Start data pre-processing")

# ----process each image to get rawpixel, histogram and label----
rawPixel = []
histogram = []
labels = []

#"Caltech256" is the dataset folder name
imagePaths = list(paths.list_images("Caltech256")) 

for (i, imagePath) in enumerate(imagePaths):
    # read each image
    image = cv2.imread(imagePath)

    # extract features to raw pixel
    size=(32,32) 
    rawPixel.append(cv2.resize(image, size).flatten())

    # extract features to histogram
    bins=(8,8,8)
    hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    histogram.append(hist.flatten())

    # extract label from the image file name
    # format of the name is "001_0001.jpg", "001" is the category id, "0001" is the image id in each category
    labels.append(imagePath.split(os.path.sep)[-1].split("_")[0] )

rawPixel = np.array(rawPixel)
histogram = np.array(histogram)
labels = np.array(labels)

# ----split data into 85% of training and 15% of testing-----
(trainRaw, testRaw, trainRawLabel, testRawLabel) = train_test_split(rawPixel, labels, test_size=0.15, random_state=42)
(trainHistogram, testHistogram, trainHistogramLabel, testHistogramLabel) = train_test_split(histogram, labels, test_size=0.15, random_state=42)

print("End data pre-processing")

# ----------------------------------------KNN-------------------------------------------

# -----change k here------
k=4   
# ---12, 14, 16, 18, 20---

print("\n")
print("Start KNN")
print("k = "+str(k))
#raw pixel
model = KNeighborsClassifier(n_neighbors=k)
model.fit(trainRaw, trainRawLabel)
acc = model.score(testRaw, testRawLabel)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))
#histogram
model = KNeighborsClassifier(n_neighbors=k)
model.fit(trainHistogram, trainHistogramLabel)
acc = model.score(testHistogram, testHistogramLabel)
print("Histogram accuracy: {:.2f}%".format(acc * 100))

#-------------------------------------Neural Network-------------------------------------
print("\n")
print("Start Neural Network")
#raw pixel
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
model.fit(trainRaw, trainRawLabel)
acc = model.score(testRaw, testRawLabel)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))
#histogram
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1)
model.fit(trainHistogram, trainHistogramLabel)
acc = model.score(testHistogram, testHistogramLabel)
print("Histogram accuracy: {:.2f}%".format(acc * 100))

#------------------------------------------SVM-------------------------------------------

# -----change kernel trick here--------
kernelTrick='poly' 
# -----'poly', 'rbf', 'linear'--------

print("\n")
print("Start SVM")
print('Kernel trick = '+str(kernelTrick))
#raw pixel
model = SVC(kernel=kernelTrick, max_iter=100,class_weight='balanced')
model.fit(trainRaw, trainRawLabel)
acc = model.score(testRaw, testRawLabel)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))
#histogram
model = SVC(kernel=kernelTrick, max_iter=100,class_weight='balanced')
model.fit(trainHistogram, trainHistogramLabel)
acc = model.score(testHistogram, testHistogramLabel)
print("Histogram accuracy: {:.2f}%".format(acc * 100))
