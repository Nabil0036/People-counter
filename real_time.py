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
from pytictoc import TicToc
from imutils.video import WebcamVideoStream as webcam
from imutils.video import FPS

#---------for database----------#
da = Database_Utils('people.db')
da.create_table()
print("table create done")
#------database end------_#
#----------------------------------------#
print("Starting Video.....")
#cap = cv2.VideoCapture(0)
vs = webcam(0).start()
print("Entry Camera found")
#------------------------------------------#
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
        da.write_to_file(img_blob,'/faces'+'/'+str(id)+'.jpg')
except:
    print("read_from_db_failed")



temp_database = f.update_temp_database_enter()
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
#------------------------------------#
path_proto = 'deploy.prototxt.txt'
path_model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#--------------------------------------#
co =0
t = TicToc()
t.tic()
fps = FPS().start()
frames_after_insertion = 100
temp_database = []
last_id = p
entered_entry = []
while True:
    fps.update()
    co+=1 
    t.toc()
    print(temp_database)
    print("frames",co)
    entered_people = len(da.read_from_db_only_entered())
    #temp_database = f.update_temp_database_enter()
    #ret, frame = cap.read()
    frame = vs.read()
    boxes = f.detect_face_dnn(net,frame,0.7)
    #cheak for if the boxes are tuple or not
    if co == frames_after_insertion:
        for temp in temp_database:
            t_id, emd,entered,entry_time, exit_time = temp
            if t_id not in entered_entry:
                da.data_entry(t_id,face,state,enty_time,"")
                entered_entry.append(t_id)
            else:
                continue



    check_tuple = type(boxes) is tuple
    if len(boxes)>=1 and not check_tuple:
        for box in boxes:
            x,y,w,h = box[0],box[1],box[2],box[3]
            tup_box = (x,y,w,h)
            if w>60 and h >60:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                try:
                    face = f.return_face(frame,tup_box)
                except:
                    cv2.imshow('Entry Camera', frame)
                    continue
                real_emd = f.face_embedding(model,face)
                if len(temp_database)==0:
                    state = 'Entered'
                    enty_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    p+=1
                    people = (p,real_emd,state,enty_time,"")
                    temp_database.append(people)
                    #da.data_entry(p,face,state,enty_time,"")
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
                            #da.data_entry(p,face,state,enty_time,"")
                        else:
                            continue


                print(len(temp_database))
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, 'People in the room: '+str(entered_people), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Entry Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.putText(frame, 'People in the room: '+str(entered_people), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Entry Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# When everything is done, release the capture
#cap.release()
cv2.destroyAllWindows() 