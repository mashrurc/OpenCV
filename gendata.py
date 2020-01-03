import cv2
from keras.models import load_model

import numpy as np
from keras.preprocessing import image

import os, os.path
DIR = 'C:/Users/Mashrur/Desktop/OpenCV/unknownfaces'

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
r=0
while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    for f in faces:
        r+=1
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "unknownfaces/face"+str(r)+".jpg"
        cv2.imwrite(FaceFileName, sub_face)
    # Show the image
    cv2.imshow('Face Crop',  im)
    key = cv2.waitKey(10)
    if key == 27: #The Esc key
        break
