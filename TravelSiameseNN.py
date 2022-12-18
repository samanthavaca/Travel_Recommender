#TravelSiameseNN.py

#STEP 1- IMPORT PACKAGES __________________________________________________
from distutils.command.build import build
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
import keras.losses
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
import umap
from tensorflow.keras.layers import *
import math
import cv2

#STEP 2- DEFINE THE MODEL PARAMETERS __________________________________________________

#Epochs: the number of iterations over the testing set
EPOCHS = 30

#Batch Size: the number of images looked over at once
BATCH_SIZE = 64

#Image Shape: the shape of the image vectors
IMG_SHAPE = (28, 28, 3)

NUM_CLASSES = 10

#STEP 3- GET DATA __________________________________________________

path_test = "./traveldataset.npz"
with np.load(path_test) as dataset:
    trainX_raw = dataset['x_train']
    print(trainX_raw)
    trainY_raw = dataset['y_train']
    print(trainY_raw)
    testX_raw = dataset['x_test']
    testY_raw = dataset['y_test']

#Normalize image pixels
trainX_raw = trainX_raw.astype("float32") / 255.0
testX_raw = testX_raw.astype("float32") / 255.0

#STEP 4- CREATE TRIPLETS __________________________________________________

#Create triplets by picking random from anchor, then picking random positive and picking random negative
def create_triplets(batch_size=BATCH_SIZE):

    while True:
        imgsA = []
        imgsP = []
        imgsN = []
        triplet_labels = []
    
        #Construct a list of the location of every image of a class
        num_classes = len(np.unique(trainY_raw))
        NUM_CLASSES = num_classes

        indices = []
        for i in range(0, num_classes):
            currentClass = np.where(trainY_raw == i)[0]
            indices.append(currentClass)

        #Form the triplets: each one having an anchor, positive, and negative image
        for number in range(batch_size):
            randomIndex = random.randint(0, len(trainX_raw) - 1)
            #Get the current image
            anchorImage = trainX_raw[randomIndex]
            anchorLabel = trainY_raw[randomIndex]
            imgsA.append(anchorImage)

            #Randomly choose positive image- an image from the same class as the anchor image
            positiveIndex = random.choice(indices[anchorLabel])
            positiveImage = trainX_raw[positiveIndex]
            positiveLabel = trainY_raw[positiveIndex]
            imgsP.append(positiveImage)

            #Randomly choose negative image- an image from a different class than the anchor image
            randomClass = random.randint(0, num_classes - 1)
            negativeIndex = random.choice(indices[randomClass])
            negativeImage = trainX_raw[negativeIndex]
            negativeLabel = trainY_raw[negativeIndex]
            imgsN.append(negativeImage)

            triplet_labels.append([anchorLabel, positiveLabel, negativeLabel])
   
        #Return image triplets and image labels
        yield ([np.array(imgsA), np.array(imgsP), np.array(imgsN)], np.array(triplet_labels))

#STEP 5- GENERATE EMBEDDINGS __________________________________________________
def embeddings_generator(input_shape, embedding_dim=48):
    inputs = Input(input_shape)
    
    #Input layer
    x = Conv2D(32, [7, 7], padding="same", activation="relu")(inputs)
    x = MaxPooling2D([2, 2])(x)
 
    x = Conv2D(64, [5, 5], padding="same", activation="relu")(x)
    x = Conv2D(64, [5, 5], padding="same", activation="relu")(x)
    x = Conv2D(64, [5, 5], padding="same", activation="relu")(x)
    x = MaxPooling2D([2, 2])(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
 
    x = Conv2D(128, [3, 3], padding="same", activation="relu")(x)
    x = Conv2D(128, [3, 3], padding="same", activation="relu")(x)
    x = Conv2D(128, [3, 3], padding="same", activation="relu")(x)
    x = MaxPooling2D([2, 2])(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
 
    x = Conv2D(256, [1, 1], padding="same", activation="relu")(x)
    x = Conv2D(256, [1, 1], padding="same", activation="relu")(x)
    x = Conv2D(512, [1, 1], padding="same", activation="relu")(x)
    x = MaxPooling2D([2, 2])(x)

    x = Flatten()(x)
    x = Dense(100, activation="relu")(x)

    model = Model(inputs, x)

    #Build model
    print(model.summary())
    return model

#STEP 6- DEFINE NEURAL NETWORKS WITH EMBEDDING MODELS __________________________________________________

anchor_images = Input(shape=IMG_SHAPE)
positive_images = Input(shape=IMG_SHAPE)
negative_images = Input(shape=IMG_SHAPE)

embeddings_model = embeddings_generator(IMG_SHAPE)

anchor_embeddings = embeddings_model(anchor_images)
positive_embeddings = embeddings_model(positive_images)
negative_embeddings = embeddings_model(negative_images)

output = Concatenate()([anchor_embeddings, positive_embeddings, negative_embeddings])

input = [anchor_images, positive_images, negative_images]

siamese_nn = Model(inputs=input, outputs=output)
siamese_nn.summary()

#STEP 7- DEFINE TRIPLET LOSS FUNCTION __________________________________________________
def euclidean_distance(vectors):
    #Separate vectors into lists
    (anchor, other) = vectors

    #Sum of squared distance between vectors
    sumSquared = K.sum(K.square(anchor - other), axis=1, keepdims = True)

    #Return the euclidean distance
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def triplet_loss(y_true, y_pred):
    anchor = y_pred[:, 0:100]
    positive = y_pred[:, 100:200]
    negative = y_pred[:, 200:300]
    margin = 1

    #Calculate distance between anchor & positive and anchor & negative
    positive_distance = euclidean_distance([anchor, positive])
    negative_distance = euclidean_distance([anchor, negative])

    #Calculate loss
    loss = tf.maximum(0.0, positive_distance - negative_distance + margin)
    return tf.reduce_mean(loss)

#STEP 8- COMPILE AND TRAIN MODEL __________________________________________________
    
siamese_nn.compile(loss=triplet_loss, optimizer="adam")

siamese_nn.fit_generator(create_triplets(), steps_per_epoch=150, epochs=EPOCHS)

#STEP 9- SAVE MODEL TO DISK FOR TESTING __________________________________________________

#Path to output directory
BASE_OUTPUT = "output"

#Derive path to serialized model with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])

#Serialize model to disk
siamese_nn.save(MODEL_PATH)

#Produce embeddings for every image in data set
model_embeddings = siamese_nn.layers[3].predict(testX_raw, verbose=1)
print(model_embeddings.shape)

#Reduce embeddings to two dimensions for visualization
reduced_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="correlation").fit_transform(model_embeddings)
print(reduced_embeddings.shape)

#Graph embeddings
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=testY_raw)
plt.show()