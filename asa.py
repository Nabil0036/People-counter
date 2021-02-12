from tensorflow.keras.models import load_model
from face_main import Database_Utils
import cv2
import numpy as np
da = Database_Utils('people.db')

data = da.read_from_db()
d = data[5]
img_blob = d[1]
id = int(d[0])
#/home/pi/Peple_counter/faces
da.write_to_file(img_blob,'tempo'+'.jpg')