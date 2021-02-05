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
from imutils.video import WebcamVideoStream as webcam
from imutils.video import FPS
from pytictoc import TicToc
#---------for database----------#
#conn = sqlite3.connect('people.db')
#c = conn.cursor()

da = Database_Utils('people.db')
da.create_table()

print("done")
#------database end------_#
#-----------------------------------------------------#
#cap = cv2.VideoCapture(1)
vs = webcam(1).start()
print("Camera found Successfully")
#-----------------------------------------------------#
f= Face_utils()
#-----------------------------------------------------#
model = load_model("facenet_keras.h5")
cascade_path = "haarcascade_frontalface_default.xml"
#-----------------------------------------------------#
#------------------------------------#
path_proto = 'deploy.prototxt.txt'
path_model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(path_proto, path_model)
#--------------------------------------#
try:
    data = da.read_from_db()
    for d in data:
        img_blob = d[1]
        id = d[0]
        #/home/pi/Peple_counter/faces
        da.write_to_file(img_blob,'/faces'+'/'+str(id)+'.jpg')
except:
    print("read_from_db_failed")


temp_database = f.update_temp_database_exit()
#---------------------------------------------------------#
#---------------------------------------------------------------------------#
co =0
t = TicToc()
t.tic()
fps = FPS().start()
while True:
    fps.update()
    co+=1 
    t.toc()
    print("frames",co)
    exited_person = len(da.read_from_db_only_exited())
    temp_database = f.update_temp_database_exit()
    #ret, frame = cap.read()
    frame = vs.read()
    #boxes = f.detect_face_haar_cascade(cascade_path,frame)
    #boxes = f.detect_face_dlib(frame)
    boxes = f.detect_face_dnn(net,frame,0.5)
    check_tuple = type(boxes) is tuple
    #print(boxes)
    if len(boxes)>=1 and not check_tuple:
        for box in boxes:
            #box = boxes[0]
            x,y,w,h = box[0],box[1],box[2],box[3]
            tup_box = (x,y,w,h)
            #print(tup_box)
            if w>60 and h >60:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face = f.return_face(frame,tup_box)
                real_emd = f.face_embedding(model,face)
                
                if len(temp_database)==0:
                    print("Not possible")
                else:
                    count =0
                    for t_d in temp_database:
                        id, emd,entered,entry_time, exit_time = t_d
                        print("ss",f.compare_embeddings(emd,real_emd))
                        if f.compare_embeddings(emd,real_emd)<12:
                            ti = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            da.change_state(id,ti)
                        else:
                            count+=1

                print(len(temp_database))
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, 'People exited: '+str(exited_person), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Exit Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.putText(frame, 'People exited: '+str(exited_person), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow('Exit Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    

# When everything is done, release the capture
fps.stop()
print("[INFO] elasped time for exit: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS for exit: {:.2f}".format(fps.fps()))
#cap.release()
cv2.destroyAllWindows() 