#coding=utf-8

import face_recognition
import cv2
import numpy as np
import sys
from dlib_detect import *
base_path = base_path = 'FaceDatabase/liqiwei'

jobs_image = cv2.imread(base_path+'/'+'linqiwei32.jpg')
obama_image = cv2.imread('obama.jpg')
unknown_image = cv2.imread('images/187.jpg')
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
obama_encoding = face_recognition.face_encodings(obama_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    jobs_encoding,
    obama_encoding
]
known_face_names = [
    "Yujun",
    "Obama"
]
# Initialize some variables
# face_locations = []
# face_encodings = []
face_names = []

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(unknown_image)
# mytemp= []
# mytemp.append(face_locations[0])
# print(mytemp)
# draw_face(unknown_image.copy(),face_locations)

for i in face_locations:
    temp = []
    temp.append(i)
    # print(temp)
    face_encodings = face_recognition.face_encodings(unknown_image, temp)
    # print(len(face_encodings))
    # See if the face is a match for the known face(s)
    # print(type(known_face_encodings))
    # print('------------------------------------')
    # print(known_face_encodings)
    # print(type(face_encodings))
    matches = face_recognition.compare_faces(known_face_encodings, np.asarray(face_encodings),tolerance=0.4)
    name = "Unknown"
    print(matches)
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        face_names.append(name)
        print(face_names)
        cv2.putText(unknown_image,name,(i[3],i[0]),cv2.FONT_HERSHEY_SIMPLEX,2,(155,155,255),2)
        # break
draw_face(unknown_image,face_locations)
# cv2.imshow('detect',unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
