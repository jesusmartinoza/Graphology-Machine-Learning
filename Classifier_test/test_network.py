# USAGE
# python test_network.py --model social_worried.model --image images/examples/test_positive.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
img = image.load_img(args["image"], target_size=(28, 28)).convert('L')
# print("Image size: " + str(np.shape(img)))
orig = image.load_img(args["image"], target_size=(28, 28))
orig = img_to_array(orig)
img = img_to_array(img).ravel()

# pre-process the image for classification
print("Image 1")
print(img)
img = img.astype("float") / 255.0
print("Image 2")
print(img)
img = np.expand_dims(img, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
print("Shape: " + str(np.shape(img)))
(notSanta, santa) = model.predict(img)[0]
print("Not Social Worried: " + str(notSanta) + " Worried: " + str(santa))

# build the label
label = "Social worried" if santa > notSanta else "Not social worried"
proba = santa if santa > notSanta else notSanta
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
color = (0, 255, 0) if santa > notSanta else (0, 0, 255)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, color, 2)

# show the output image
cv2.imwrite("result.png", output)
#cv2.imshow("Output", output)
#cv2.waitKey(0)