#coding=utf-8

import face_recognition
import cv2
import numpy as np
import sys
from dlib_detect import *
jobs_image = face_recognition.load_image_file("obama.jpg")
obama_image = face_recognition.load_image_file("obama.jpg")
unknown_image = face_recognition.load_image_file("obama1.jpg")
file = open('jobs_encodings.txt','w+')
locations = detect_image(jobs_image)
print(locations)
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
print(jobs_encoding)
file.write(str(jobs_encoding))
file.close()
np.savetxt('code.txt',jobs_encoding)
obama_encoding = face_recognition.face_encodings(obama_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
print(type(unknown_encoding))
results = face_recognition.compare_faces([jobs_encoding, obama_encoding], unknown_encoding )
labels = ['jobs', 'obama']

print('results:'+str(results))

for i in range(0, len(results)):
    if results[i] == True:
        print('The person is:'+labels[i])
