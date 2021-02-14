import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
from imutils.video import FPS
import os
import time
from mtcnn.mtcnn import MTCNN
from pytictoc import TicToc

#-----------------------------------------------------#
f = Face_utils()
#-----------------------------------------------------#
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#
#-----------For dnn face detection---------------------------#
path_proto = 'deploy.prototxt.txt'
path_model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#------------------------------------------------------------#
#----------------------for single picture--------------#
t = TicToc()
detector = MTCNN()
faces = os.listdir("Subject01")
b_path = 'Subject01/'
elapsed_time = 0
count = 0
for face in faces:
    if face.startswith('T'):
        continue
    print(face)
    full = b_path+face
    image = cv2.imread(full)
    image = cv2.resize(image, (500, 500))
    t0 = time.clock()
    # boxes = f.detect_face_dnn(net, image, 0.5)
    # boxes = f.detect_face_haar_cascade(cascade_path, image)  # Haar cascade
    # boxes = f.detect_face_mtcnn(detector, image)
    boxes = f.detect_face_dlib(image)
    t1 = time.clock() - t0
    check_tuple = type(boxes) is tuple
    if boxes == [] or check_tuple:
        continue
    
    elapsed_time +=t1
    count+=1
    
    
    print(boxes)
    # time.sleep(1)
    if len(boxes) >= 1 and not check_tuple:
        box = boxes[0]
        x, y, w, h = box[0], box[1], box[2], box[3]
        tup_box = (x, y, w, h)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.waitKey(1)
        #cv2.imshow("hello", image)
        cv2.imwrite("test/"+face, image)
        # time.sleep(2)
print(elapsed_time)
print(count)