#!/usr/bin/python

import sys, cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
webcam = cv2.VideoCapture(0)

if webcam.isOpened(): 
    rval, frame = webcam.read()
else:
    rval = False
 
while rval:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('frame',frame)        
    rval, frame = webcam.read()
    key = cv2.waitKey(10)

    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break

cv2.destroyAllWindows()
