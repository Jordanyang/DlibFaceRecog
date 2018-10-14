#coding=utf-8
import os
import cv2
# import numpy as np

base_path = 'newFaceDatabase/'

picture_list = os.listdir(base_path)
print(picture_list)

def process(img):
    """

    :param img: An image (as a numpy array) to image processing
    :return:
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img)
    y = cv2.equalizeHist(y)
    img = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return img
for picture in picture_list:
    target_path = base_path+'/'+str(picture)
    print(target_path)
    target_image = cv2.imread(target_path,1)
    target_image = cv2.resize(target_image,(128,128),interpolation=cv2.INTER_CUBIC)
    # target_image = cv2.GaussianBlur(target_image, (5, 5), 0)
    target_image = process(target_image)
    cv2.imwrite(target_path,target_image)