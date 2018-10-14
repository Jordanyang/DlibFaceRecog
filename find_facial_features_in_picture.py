import cv2
import dlib
import numpy as np
import sys
import matplotlib.pyplot as plt
SCALE_FACTOR = 8 # 图像的放缩比

PREDICTOR_PATH = "model/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise "TooManyFaces"
    if len(rects) == 0:
        raise "NoFaces"

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s

def annotate_landmarks(im, landmarks):
    '''
    人脸关键点，画图函数
    '''
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im
im1, landmarks1 = read_im_and_landmarks('jordan0.jpg')  # 底图
im1 = annotate_landmarks(im1, landmarks1)

cv2.imshow('hello',im1)
cv2.waitKey(0)
cv2.destroyAllWindows()

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
# 人脸对齐函数
def face_Align(Base_path,cover_path):
    im1, landmarks1 = read_im_and_landmarks(Base_path)  # 底图
    im2, landmarks2 = read_im_and_landmarks(cover_path)  # 贴上来的图

    if len(landmarks1) == 0 & len(landmarks2) == 0 :
        # raise ImproperNumber("Faces detected is no face!")
        pass
    if len(landmarks1) > 1 & len(landmarks2) > 1 :
        # raise ImproperNumber("Faces detected is more than 1!")
        pass

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    warped_im2 = warp_im(im2, M, im1.shape)
    return warped_im2

FEATHER_AMOUNT = 29  # 匹配的时候，特征数量，现在是以11个点为基准点  11  15  17

Base_path = 'jordan0.jpg'
cover_path = 'jordan1.jpg'
warped_mask = face_Align(Base_path,cover_path)
img123 = cv2.imread(Base_path)
img456 = cv2.imread(cover_path)
cv2.imshow('original face',img123)
cv2.imshow('to align face',img456)
cv2.imshow('face align',warped_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()