#coding=utf-8
import cv2
import numpy as np

img = cv2.imread('FaceDatabase/jordan/jordan12.jpg')
# cv2.imshow('original',img)
# blur = cv2.medianBlur(img,5)
# cv2.imshow('blur',blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# b,g,r = cv2.split(img)
#
# b = cv2.equalizeHist(b)
# g = cv2.equalizeHist(g)
# r = cv2.equalizeHist(r)
#
# img = cv2.merge((b,g,r))
# cv2.imshow('equalizehist',img)



# create a CLAHE object (Arguments are optional).
# 不知道为什么我没好到createCLAHE 这个模块
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# b,g,r = cv2.split(img)
# b = clahe.apply(b)
# g = clahe.apply(g)
# r = clahe.apply(r)
# img = cv2.merge((b,g,r))
# cv2.imshow('equalizehist',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
y,cr ,cb= cv2.split(img)
y = cv2.equalizeHist(y)
img = cv2.merge((y,cr,cb))
img = cv2.cvtColor(img,cv2.COLOR_YCrCb2BGR)
cv2.imshow('hello',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

