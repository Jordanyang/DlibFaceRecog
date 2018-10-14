import cv2
import face_recognition

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("7_Cheering_Cheering_7_60.jpg")

# Find all the faces in the image using a pre-trained convolutional neural network.
# This method is more accurate than the default HOG model, but it's slower
# unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
# this will use GPU acceleration and perform well.
# See also: find_faces_in_picture.py
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,model='cnn')

print("I found {} face(s) in this photograph.".format(len(face_locations)))

# for face_location in face_locations:
#
#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
#
#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     cv2.imshow('face',face_image)
#     cv2.waitKey(0)

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
    cv2.imshow('detect face',img)
if __name__=='__main__':
    draw_face(image, face_locations)
    cv2.waitKey(0)
