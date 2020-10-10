import dlib
import cv2
from face_main import Face_utils,Database_Utils
import numpy as np
from tensorflow.keras.models import load_model
import os 
import sqlite3
import array
import pickle
from datetime import datetime


#---------for database----------#

da = Database_Utils()
da.create_table()
print("table create done")
#------database end------_#
#-----------------------------------------------------#
print("Starting Video.....")
cap = cv2.VideoCapture(0)
#-----------------------------------------------------#
f= Face_utils()
#-----------------------------------------------------#
print("Loading model.....")
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#

try:
    a = da.read_last_entry()
    print("aaassss",a)
except:
    a=0
    print("aaassss",a)


i=0
try:
    data = da.read_from_db()
    for d in data:
        img_blob = d[1]
        id = d[0]
        #/home/pi/Peple_counter/faces
        da.write_to_file(img_blob,'/home/pi/People_counter/faces'+'/'+str(id)+'.jpg')
except:
    print("read_from_db_failed")

def update_temp_database():
    try:
        temp_database=[]
        data = da.read_from_db_only_entered()
        for d in data:
            with open('temp.jpg','wb') as file:
                file.write(d[1])
            img = cv2.imread('temp.jpg')
            x = f.face_embedding(model,img)
            a_,b_,c_,d_,e_ = d[0],x,d[2],d[3],d[4]
            single = (a_,b_,c_,d_,e_)
            temp_database.append(single)
    except:
        temp_database = []

    return temp_database


temp_database = update_temp_database()
p=a
#---------------------------------------------------------#
# image = cv2.imread('/home/pi/Peple_counter/nabil2.jpg')
# faces = f.detect_face_haar_cascade('/home/pi/Peple_counter/haarcascade_frontalface_default.xml',image)
# box = faces[0]
# x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
# print(x1,y2,x2,y2)
# roi = f.return_face(image,box)
# nab_emd = f.face_embedding(model, roi)
#---------------------------------------------------------------------------#
while True:
    temp_database = update_temp_database()
    ret, frame = cap.read()
    boxes = f.detect_face_haar_cascade(cascade_path,frame)
    #cheak for if the boxes are tuple or not
    check_tuple = type(boxes) is tuple
    if len(boxes)>=1 and not check_tuple:
        for box in boxes:
            x,y,w,h = box[0],box[1],box[2],box[3]
            tup_box = (x,y,w,h)
            if w>60 and h >60:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face = f.return_face(frame,tup_box)
                real_emd = f.face_embedding(model,face)
                if len(temp_database)==0:
                    state = 'Entered'
                    enty_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    p+=1
                    people = (p,real_emd,state,enty_time,"")
                    temp_database.append(people)
                    da.data_entry(p,face,state,enty_time,"")
                else:
                    count =0
                    for t_d in temp_database:
                        id, emd,entered,entry_time, exit_time = t_d
                        print("ss",f.compare_embeddings(emd,real_emd))
                        if f.compare_embeddings(emd,real_emd)<12:
                            break
                        else:
                            count+=1
                        if count==len(temp_database):
                            state = 'Entered'
                            enty_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            p+=1
                            people = (p,real_emd,state,entry_time,"")
                            temp_database.append(people)
                            da.data_entry(p,face,state,enty_time,"")



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