"""
code originated from https://youtu.be/Ax6P93r32KU
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:50:23 2021

@author: conle
"""

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY =  "C:/Users/conle/Documents/AiClub/AIClub"
CATEGORIES = ["jpgMask", "jpgFace"]

SIZE = 250

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    #combine the main directory path with specific data folder
    path = os.path.join(DIRECTORY, category)
    #iterate through all images in the folder
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
        #simply loads image to size SIZExSIZE and save to omage
    	image = load_img(img_path, target_size=(SIZE, SIZE))
        #image to array instead of multidimensional
    	image = img_to_array(image)
        #need since we are using mobile nets
    	image = preprocess_input(image)

    	data.append(image)
        #with mask 0 and without 1
    	labels.append(category)

# perform one-hot encoding on the labels  dont thinkg this is necessary
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
#data needs to be in numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

#this seperates the data randomly into test and train data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

#-------  end of data pre-processing  --------


# construct the training image generator for data augmentation
#this creates more data imag, basically by distorting the current data. (rotating, zoomin, etc)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


"""Mobile nets will generate two types of models
Type 1: head model
    whos output will be basing into     

Type 2:Base model
    made for image networks (weights = "imagenet")
    """



# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(SIZE, SIZE, 3)))
                    #3 because RGB

# construct the head of the model that will be placed on top of the
# the base model
#passing the basemodel output as first parameter
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
#relu is used for non-linear use cases (for images)
headModel = Dense(128, activation="relu")(headModel)
#Dropout-->avoid overfitting
headModel = Dropout(0.5)(headModel)
#2 because there are two options, softmax for binary answer as well
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
#accepts input and output,   
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#opt is used for images, track accuracy
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
#aug.flow adds extra data
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save the model for later uses in other programs
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
