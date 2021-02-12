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
from multiprocessing import Process
from threading import Thread


#---------for database----------#
#------------------------------------------#
f= Face_utils()
da = Database_Utils('people.db')
da.create_table()
print("table create done")
#------database end------_#
#----------------------------------------#
print("Starting Video.....")
vs_entry = webcam(0).start()
vs_exit = webcam(1).start()
print("Entry Camera found")
print("Loading model.....")
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
print("Model Load Done!")
#-----------------------------------------------------#
#-----------------Dummy Recognition----------------------------------------#
image = cv2.imread('dummy.jpg')
faces = f.detect_face_haar_cascade('haarcascade_frontalface_default.xml',image)
box = faces[0]
x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
print(x1,y2,x2,y2)
roi = f.return_face(image,box)
nab_emd = f.face_embedding(model, roi)
#---------------------------------------------------------------------------#
try:
    a = da.read_last_entry()
except:
    a=0

i=0
try:
    data = da.read_from_db_only_entered()
    entered_ids = [x[0] for x in data]
    #print("sssssssssssss",data[0])
    print("---Entered IDs------------",entered_ids)
    temp_database = []
    for d in data:
        with open('temp.jpg','wb') as file:
            file.write(d[1])
        img = cv2.imread('temp.jpg')
        x = f.face_embedding(model,img)
        single = (d[0],x,d[2],d[3],d[4],img)
        temp_database.append(single)
        print("done")
except:
    temp_database = []
    entered_ids = []
    print("read_from_db_failed")
p=a
path_proto = 'deploy.prototxt.txt'
path_model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#--------------------------------------#
co =0
t = TicToc()
fps = FPS().start()
frames_after_insertion = 20

def entry_func(frame):
    global p
    global temp_database
    entered_people = []
    boxes = f.detect_face_dnn(net,frame,0.7)
    entered_people = list(filter(lambda x:x[2]=='Entered',temp_database))
    print("kuki",entered_people)
    #cheak for if the boxes are tuple or not
    check_tuple = type(boxes) is tuple
    if len(boxes)>=1 and not check_tuple:
        for box in boxes:
            x,y,w,h = box[0],box[1],box[2],box[3]
            tup_box = (x,y,w,h)
            
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
                people = (p,real_emd,state,enty_time,"",face)
                temp_database.append(people)
            else:
                count =0
                for t_d in temp_database:
                    id, emd,entered,entry_time, exit_time, ui = t_d
                    if f.compare_embeddings(emd,real_emd)<12 and entered=='Entered':
                        break
                    else:
                        count+=1
                    if count==len(temp_database):
                        state = 'Entered'
                        enty_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        p+=1
                        people = (p,real_emd,state,entry_time,"",ui)
                        temp_database.append(people)
                    else:
                        continue
            # print(entered_people)
            cv2.putText(frame, 'People in the room: '+str(len(entered_people)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Entry Camera', frame)
    else:
        cv2.putText(frame, 'People in the room: '+str(len(entered_people)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Entry Camera', frame)

def exit_func(frame):
    global temp_database
    global p
    global exited
    exited_person = []
    boxes = f.detect_face_dnn(net,frame,0.5)
    check_tuple = type(boxes) is tuple
    #print(boxes)
    exited_person = list(filter(lambda x:x[2]=='Exited',temp_database))
    if len(boxes)>=1 and not check_tuple:
        for box in boxes:
            #box = boxes[0]
            x,y,w,h = box[0],box[1],box[2],box[3]
            tup_box = (x,y,w,h)
            #print(tup_box)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            face = f.return_face(frame,tup_box)
            real_emd = f.face_embedding(model,face)
            
            if len(temp_database)==0:
                print("Not possible")
            else:
                count =0
                for w,t_d in enumerate(temp_database):
                    id, emd,entered,entry_time, exit_time, _ = t_d
                    if f.compare_embeddings(emd,real_emd)<12 and entered=='Entered':
                        ti = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        t_d_l = list(t_d)
                        t_d_l[2] = 'Exited'
                        t_d_l[4] = ti
                        t_d_t = tuple(t_d_l)
                        temp_database[w] = t_d_t
                        break
                    else:
                        count+=1
                    if count == len(temp_database):
                        print("Something went wrong")
            cv2.putText(frame, 'People exited: '+str(exited), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Exit Camera', frame)
    else:
        cv2.putText(frame, 'People exited: '+str(exited), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Exit Camera', frame)
        
t.tic()
exited = 0
print("Temp database before exec",temp_database[0])
if __name__=='__main__':
    while True:
        fps.update()
        co+=1
        frame_entry = vs_entry.read()
        frame_exit = vs_exit.read()

        entry_func(frame_entry)
        exit_func(frame_exit)
        # t1 = Thread(target=entry_func,args=(frame_entry,))
        # t1.daemon = True
        # t2 = Thread(target=exit_func,args=(frame_exit,))
        # t2.daemon = True

        # t1.start()
        # t2.start()
        # t1.join()
        # t2.join()
        if co == frames_after_insertion:
            exited = da.sync_database(temp_database,entered_ids,exited)
            co =0
        t.toc()
        print(co)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))