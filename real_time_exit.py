import dlib
import cv2
from face_main import Face_utils
import numpy as np
from tensorflow.keras.models import load_model
import os 
import sqlite3
import array
import pickle

#---------for database----------#
conn = sqlite3.connect('people.db')
c = conn.cursor()
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS my_table(id REAL, image BLOB)')

def data_entry(id, img):
    cv2.imwrite("/home/pi/Peple_counter/hudai.png",img)
    with open("/home/pi/Peple_counter/hudai.png","rb") as file:
        pic = file.read()

    c.execute("INSERT INTO my_table (id,image) VALUES (?, ?)",(id,pic))
    conn.commit()

create_table()

print("done")
#------database end------_#
#-----------------------------------------------------#
cap = cv2.VideoCapture(1)
#-----------------------------------------------------#
f= Face_utils()
#-----------------------------------------------------#
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#
def read_from_db():
    c.execute('SELECT * FROM my_table')
    data = c.fetchall()
    return data
data = read_from_db()

print(data)
temp_database = []
p=0
#---------------------------------------------------------#
image = cv2.imread('/home/pi/Peple_counter/nabil2.jpg')
faces = f.detect_face_haar_cascade('/home/pi/Peple_counter/haarcascade_frontalface_default.xml',image)
box = faces[0]
x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
print(x1,y2,x2,y2)
roi = f.return_face(image,box)
nab_emd = f.face_embedding(model, roi)
#---------------------------------------------------------------------------#
while True:
    ret, frame = cap.read()
    boxes = f.detect_face_haar_cascade(cascade_path,frame)
    check_tuple = type(boxes) is tuple
    #print(boxes)
    if len(boxes)>=1 and not check_tuple:
        box = boxes[0]
        x,y,w,h = box[0],box[1],box[2],box[3]
        tup_box = (x,y,w,h)
        #print(tup_box)
        if w>60 and h >60:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face = f.return_face(frame,tup_box)
            real_emd = f.face_embedding(model,face)
            
            if len(temp_database)==0:
                state = 'Entered'
                people = (p,real_emd,state)
                temp_database.append(people)
                data_entry(p,face)
            else:
                count =0
                for t_d in temp_database:
                    id, emd,entered = t_d
                    print("ss",f.compare_embeddings(emd,real_emd))
                    if f.compare_embeddings(emd,real_emd)<12:
                        break
                    else:
                        count+=1
                    if count==len(temp_database):
                        state = 'Entered'
                        people = (p,real_emd,state)
                        temp_database.append(people)


            print(len(temp_database))
        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows() 