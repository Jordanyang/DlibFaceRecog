#coding=utf-8

import face_recognition
import cv2
import numpy as np
import sys
from dlib_detect import *

jordan_image = []
linqiwei_image = []
yujun_image = []
jordan_encoding = []
linqiwei_encoding = []
yujun_encoding = []
for i in range(0,33):
    target = 'jordan'+str(i)+'.jpg'
    image_path = 'FaceDatabase/'+'jordan/'+target
    # print(image_path)
    jordan_image.append(cv2.imread(image_path))

for i in range(0,33):
    target = 'linqiwei'+str(i)+'.jpg'
    image_path = 'FaceDatabase/'+'liqiwei/'+target
    # print(image_path)
    linqiwei_image.append(cv2.imread(image_path))

for i in range(0,33):
    target = 'yujun' + str(i) + '.jpg'
    image_path = 'FaceDatabase/' + 'yujun/' + target
    # print(image_path)
    yujun_image.append(cv2.imread(image_path))
# print(len(jordan_image))
# print(jordan_image[0])
j = 0
for i in range(0,len(jordan_image)):
    temp = jordan_image[i]
    temp1 = face_recognition.face_encodings(temp)
    # print(len(temp1))
    if len(temp1) == 1:
        j+=1
        jordan_encoding.append(temp1[0])
    else:
        j = j

for i in range(0,len(linqiwei_image)):
    temp = linqiwei_image[i]
    temp1 = face_recognition.face_encodings(temp)
    # print(len(temp1))
    if len(temp1) == 1:
        j+=1
        linqiwei_encoding.append(temp1[0])
    else:
        j = j

for i in range(0,len(yujun_image)):
    temp = yujun_image[i]
    temp1 = face_recognition.face_encodings(temp)
    # print(len(temp1))
    if len(temp1) == 1:
        j+=1
        yujun_encoding.append(temp1[0])
    else:
        j = j
known_face_encodings = []
for i in range(0,len(jordan_encoding)):
    known_face_encodings.append(jordan_encoding[i])
for i in range(0,len(linqiwei_encoding)):
    known_face_encodings.append(linqiwei_encoding[i])
for i in range(0,len(yujun_encoding)):
    known_face_encodings.append(yujun_encoding[i])

known_face_names = []
for i in range(0,len(jordan_encoding)):
    known_face_names.append('yanghongyu')
for i in range(0,len(linqiwei_encoding)):
    known_face_names.append('linqiwei')
for i in range(0,len(yujun_encoding)):
    known_face_names.append('yujun')

def Face_Recognition(img):
    '''

    :param img: An image (as a numpy array) to recognition face
    :return:
    '''
    # # Initialize some variables
    face_names = []
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(img)
    for i in face_locations:
        temp = []
        temp.append(i)
        face_encodings = face_recognition.face_encodings(img, temp)
        matches = face_recognition.compare_faces(known_face_encodings, np.asarray(face_encodings),tolerance=0.4)
        name = "Unknown"
        # print(matches)
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            face_names.append(name)
            print(face_names)
            cv2.putText(img,name,(i[3],i[0]),cv2.FONT_HERSHEY_SIMPLEX,2,(155,155,155),2)
            # break
    draw_face(img,face_locations)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def face_video_recog(input_video_path, output_video_path):
    """

    :param input_video_path: the video path for detect
    :param output_video_path: the result path for detect
    :return:
    """
    vid = cv2.VideoCapture(input_video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # 视频的编码
    Video_FourCC = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # 视频的帧率
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    # 视频的宽度和高度
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_video_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_video_path), type(Video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_video_path, Video_FourCC, video_fps, video_size)
    while True:
        flag, frame = vid.read()
        if not flag:
            break
        Face_Recognition(frame)
        if isOutput:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    out.release()

def video_test():
    input_video_path = 'face.mp4'
    output_video_path = 'face_detect_recognition.avi'
    face_video_recog(input_video_path, output_video_path)

def batch_picture_test():
    test_base_path = 'images/'
    for i in range(0, 10):
        image_path = test_base_path + str(i) + '.jpg'
        print(image_path)
        img = cv2.imread(image_path)
        Face_Recognition(img)
        cv2.imwrite('results/' + str(i) + '.jpg', img)
if __name__=='__main__':
    batch_picture_test()
    # video_test()




