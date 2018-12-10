import matplotlib
import matplotlib.pyplot as plt

import numpy as np # linear algebra
from keras import models
from keras import layers
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tensorflowjs as tfjs

from imutils import paths
import argparse
import random
import cv2
import os

# Initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 50
INIT_LR = 1e-3
BS = 4

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# Gather images from path
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    img = image.load_img(imagePath, target_size=(28, 28)).convert('L')
    # print("Image size: " + str(np.shape(img)))
    img = img_to_array(img).ravel()
    # print(np.shape(img))
    data.append(img)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "m_worried" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("Shape of all data: " + str(np.shape(data)))

# Split data into test and train
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY), sep = " -- ")

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

print(np.shape(trainY))

# Create model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='mean_squared_error',
                metrics=['accuracy'])

history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BS)

print("[INFO] serializing network...")
# Load Keras model
model.save(args["model"])
tfjs.converters.save_keras_model(model, 'model')

# Evaluate loaded model on test data:
score = model.evaluate(testX, testY, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), history.history['loss'], label="train_loss")
plt.plot(np.arange(0, N), history.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, N), history.history['acc'], label="train_acc")
plt.plot(np.arange(0, N), history.history['val_acc'], label="val_acc")
plt.title("Training loss and accuracy on handwritten M")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])