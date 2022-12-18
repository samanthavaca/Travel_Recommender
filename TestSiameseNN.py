#TestSiameseNN.py

#Import packages
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.layers import Lambda

#Shape of images
IMG_SHAPE = (28, 28, 1)

#Specify batch size and epochs
BATCH_SIZE = 64
EPOCHS = 10

#Path to output directory
BASE_OUTPUT = "output"

#Derive path to serialized model with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

#Define function to make pairs
def make_pairs(images, labels):
    pairImages = []
    pairLabels = []

    #Get the number of classes
    numClasses = len(np.unique(labels))
    
    #Build list of indexes for each class label
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    #Positive and negative pair generation
    for idxA in range(len(images)):
        #Get current image and label
        currentImage = images[idxA]
        label = labels[idxA]

        #Randomly pick image from the same class
        idxB = np.random.choice(idx[label])
        posImage = images[idxB]

        #Append positive pair to lists
        pairImages.append([currentImage, posImage])

        #Indicate positive pair
        pairLabels.append([1])
        
        #Randomly get image not from same class
        negIdx = np.where(labels != label)[0]
        negImage = images[np.random.choice(negIdx)]

        #Append negative pair to lists
        pairImages.append([currentImage, negImage])
        pairLabels.append([0])

    #Return list of images pairs and labels
    return (np.array(pairImages), np.array(pairLabels))

#Define the euclidean distance function
def euclidean_distance(vectors):
    #Separate vectors into lists
    (featsA, featsB) = vectors

    #Sum of squared distance between vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims = True)

    #Return the euclidean distance
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

#Define the function to accept the training history from the model
def plot_training(H, plotPath):
    #Build a plot that saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="trian_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label = "train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

#Construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of testing images")
args = vars(ap.parse_args())

#Grab test dataset images and randomly generate 10 image pairs
testImagePaths = list(list_images(args["input"]))
pairs = np.random.choice(testImagePaths, size=(10, 2))

#Define triplet loss function
def triplet_loss(y_true, y_pred):
    anchor = y_pred[:, 0:100]
    positive = y_pred[:, 100:200]
    negative = y_pred[:, 200:300]

    #Compute distance between anchor & positive and anchor & negative
    pos_dist = K.sum(K.abs(anchor - positive), axis=1)
    neg_dist = K.sum(K.abs(anchor - negative), axis=1)

    #Calculate loss
    probs = K.softmax([pos_dist, neg_dist], axis=0)

    return K.mean(K.abs(probs[0]) + K.abs(1.0 - probs[1]))

#Load the model from disk
print("Loading the model")
model = load_model(MODEL_PATH, custom_objects={'triplet_loss': triplet_loss})

#Compare for similarity
for (i, (pathA, pathB)) in enumerate(pairs):
    #Load images
    imageA = cv2.imread(pathA)
    imageB = cv2.imread(pathB)

    #Resize the images
    resizedImageA = cv2.resize(imageA, (28, 28), interpolation = cv2.INTER_AREA)
    resizedImageB = cv2.resize(imageB, (28, 28), interpolation = cv2.INTER_AREA)

    #Create copy of images for visualization
    origA = imageA.copy()
    origB = imageB.copy()

    #Add a channel dimension to both images
    resizedImageA = np.expand_dims(resizedImageA, axis=-1)
    resizedImageB = np.expand_dims(resizedImageB, axis=-1)

    #Add a batch dimension to both images
    resizedImageA = np.expand_dims(resizedImageA, axis=0)
    resizedImageB = np.expand_dims(resizedImageB, axis=0)

    #Scale pixels to range [0, 1]
    resizedImageA = resizedImageA / 255.0
    resizedImageB = resizedImageB / 255.0

    #Use model to make predictions on image pair
    predanchor = model.layers[3].predict(resizedImageA)
    predpositive = model.layers[3].predict(resizedImageB)
    
    similarityPositive = euclidean_distance([predanchor, predpositive])

    positiveSame = "Different"

    if (similarityPositive[0][0] <= 1):
        positiveSame = "Same"
    else:
        positiveSame = "Different"

    #Initialize figure
    fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
    plt.suptitle("Similarity: {:.2f}\nPrediction: {}".format(similarityPositive[0][0], positiveSame))

    #Show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(origA, cmap=plt.cm.gray)
    plt.axis("off")

    #Show second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(origB, cmap=plt.cm.gray)
    plt.axis("off")

    #Show the plot
    plt.show()