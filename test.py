import cv2 
from face_main import Face_utils
from tensorflow.keras.models import load_model
import os

base_path = os.getcwd()
model_path = os.path.join(base_path,'facenet_keras.h5')
f = Face_utils()
model = load_model(model_path)
image = cv2.imread('/home/pi/Peple_counter/nabil2.jpg')
faces = f.detect_face_haar_cascade('/home/pi/Peple_counter/haarcascade_frontalface_default.xml',image)
box = faces[0]
x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
print(x1,y2,x2,y2)
roi = f.return_face(image,box)

cv2.imshow("image",roi)
cv2.waitKey(0)
cv2.destroyAllWindows()