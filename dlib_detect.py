# -*- coding: utf-8 -*-

# 检测人脸
import face_recognition
import cv2
import os

def detect_image(img,model="hog"):
    '''
    :param img: An image (as a numpy array)
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    '''
    face_locations = face_recognition.face_locations(img,model=model)
    # print(face_locations)
    return face_locations

def detect_video(video_path,output_video_path):
    """"
    :param video_path: the video path for detect
    :param output_video_path: the result path for detect
    :return:
    """
    vid = cv2.VideoCapture(video_path)
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
        face_locations = face_recognition.face_locations(frame)
        # 遍历每个人脸，并标注
        faceNum = len(face_locations)
        for i in range(0, faceNum):
            top = face_locations[i][0]
            right = face_locations[i][1]
            bottom = face_locations[i][2]
            left = face_locations[i][3]
            start = (left, top)
            end = (right, bottom)
            color = (55, 255, 155)
            thickness = 3
            cv2.rectangle(frame, start, end, color, thickness)
        if isOutput:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    out.release()

def draw_face(img,face_locations):
    """

    :param img: 待标注的原始人脸图像
    :param locations: 检测出的人脸坐标位置列表
    :return:
    """
    # 遍历每个人脸，并标注
    faceNum = len(face_locations)
    for i in range(0, faceNum):
        top = face_locations[i][0]
        right = face_locations[i][1]
        bottom = face_locations[i][2]
        left = face_locations[i][3]
        start = (left, top)
        end = (right, bottom)
        # 框的颜色
        color = (55, 255, 155)
        # 线的粗细
        thickness = 3
        # 画矩形框
        cv2.rectangle(img, start, end, color, thickness)
    # cv2.imshow('detect face',img)

def get_ROI(img,location):
    # 根据上、右、下、左的坐标排列方式选取ROI
    # opencv numpy array 按照从上到下，从左到右的顺序格式提取ROI区域
    top,right,bottom,left = location
    return img[top:bottom,left:right].copy()
if __name__ == '__main__':
    input_video_path = 'face.mp4'
    output_video_path = 'face_detect.avi'
    # detect_video(input_video_path, output_video_path)
    img = cv2.imread('NewFace/jordan2.jpg')
    if not img:
        raise IOError('could not open file,please check file path')
    location = detect_image(img,model='cnn')
    print(location)
    #  Get ROI Face Image
    face_image = get_ROI(img,location[0])
    draw_face(img, location)
    cv2.imshow('detect', img)
    cv2.imshow('ROI', face_image)
    cv2.imwrite('jordan1.jpg',face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
