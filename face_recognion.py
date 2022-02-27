import cv2 as cv
import numpy as np
import os
haar_cascade = cv.CascadeClassifier(
    'C:\\Python39\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

people = []
for i in os.listdir(r'C:\Users\ADB\Desktop\Study Material\IMAGEPROCESSINGAF\faceRecognition\train'):
    people.append(i)

#features = np.load('features.npy')
#labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('faceRecognition/face_trained.yml')

img = cv.imread(
    r'C:\Users\ADB\Desktop\Study Material\IMAGEPROCESSINGAF\faceRecognition\train\mindy_kaling\httpimagesnymagcomimagesdailymindykalingxjpg.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]}, confidence = {confidence}')

    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (35, 16, 127), thickness=3)


cv.imshow('face', img)
cv.waitKey(0)
