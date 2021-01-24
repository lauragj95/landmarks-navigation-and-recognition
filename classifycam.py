# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:48:34 2018

@author: laura.galvez.jimenez
"""

# USAGE
# python classifycam.py --model nodos.model --labelbin nodos.pickle 

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2


def get_video(video):
    cap = cv2.VideoCapture(video)

    while True:
        ret,frame = cap.read()
        if not ret:
           return

        yield frame


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")

args = vars(ap.parse_args())
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# load the image
for frame in get_video("20181106_201425.mp4"):
    
    output = imutils.resize(frame, width=400)
    # pre-process the image for classification
    image = cv2.resize(frame, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network and the multi-label
    # binarizer
    print("[INFO] loading network...")

    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    # loop over the indexes of the high confidence class labels
        # build the label and draw the label on the image
    label = mlb.classes_[np.argmax(proba)]
    
    if (np.amax(proba)>0.99):
        label = "{}: {:.2f}%".format(mlb.classes_[np.argmax(proba)], np.amax(proba) * 100)
        cv2.putText(output, label, (10, 40),	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))
            
    # show the output imageq
    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()