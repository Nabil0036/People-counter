import cv2
import dlib
import numpy as np
from imutils import face_utils
import sqlite3

class Face_utils:
    @staticmethod
    def detect_face(face_path):
        detector = dlib.get_frontal_face_detector()
        img = dlib.load_rgb_image(face_path)
        dets = detector(img,1)
        boxes=[]
        for i, d in enumerate(dets):
            (x,y,w,h)= face_utils.rect_to_bb(d)
            box = (x,y,w,h)
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_haar_cascade(haar_cascade_path,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(haar_cascade_path)

        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if (len(faces)==0):
            return None, None
        return faces

    @staticmethod
    def detect_face_dlib(image):
        detector = dlib.get_frontal_face_detector()
        #img = dlib.load_rgb_image(face_path)
        dets = detector(image,1)
        boxes=[]
        for i, d in enumerate(dets):
            (x,y,w,h)= face_utils.rect_to_bb(d)
            box = (x,y,w,h)
            boxes.append(box)
        return boxes

    @staticmethod
    def detect_face_dnn(net,image,con=0.9):
        boxes = []
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        (h, w) = image.shape[:2]
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < con:
                continue
            box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            x1_,y1_,x2_,y2_ = box[0],box[1], box[2], box[3]
            w_ = x2_ - x1_
            h_ = y2_ - y1_
            box = [x1_,y1_,w_,h_]
            boxes.append(box)
        return boxes
    
    @staticmethod
    def update_temp_database_exit():
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
    
    @staticmethod
    def update_temp_database_enter():
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

    @staticmethod
    def return_face(image,box):
        x, y, w, h = box
        roi = image[y:y+h,x:x+w]
        roi_resize = cv2.resize(roi,(160,160))
        return roi_resize

    @staticmethod
    def face_embedding(model,roi):
        roi = np.array(roi)
        face_pix = roi.astype('float32')
        mean, std = face_pix.mean(), face_pix.std()
        face_pix = (face_pix-mean) / std
        sample = np.expand_dims(face_pix,axis=0)
        emd = model.predict(sample)
        return emd[0]
    
    @staticmethod
    def compare_embeddings(emd1,emd2):
        return np.linalg.norm(emd1-emd2)

    @staticmethod
    def draw_text(image,text,origin):
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2 , cv2.LINE_AA)


class Database_Utils:
    def __init__(self,db):
        self.db = db
        self.conn = sqlite3.connect(self.db)
        self.c = self.conn.cursor()

    def create_table(self):
        self.c.execute('CREATE TABLE IF NOT EXISTS my_table(id REAL, image BLOB, entry_state TEXT, entry_time TEXT, exit_time TEXT)')

    def data_entry(self,id, img,entry_state="",entry_time="",exit_time=""):
        cv2.imwrite("hudai.png",img)
        with open("hudai.png","rb") as file:
            pic = file.read()
        self.c.execute("INSERT INTO my_table (id,image,entry_state,entry_time,exit_time) VALUES (?, ?, ?, ?, ?)",(id,pic,entry_state,entry_time,exit_time))
        self.conn.commit()

    def read_from_db(self):
        self.c.execute('SELECT * FROM my_table')
        data = self.c.fetchall()
        return data

    def read_from_db_only_entered(self):
        self.c.execute('SELECT * FROM my_table WHERE entry_state="Entered"')
        data = self.c.fetchall()
        return data

    def read_from_db_only_exited(self):
        self.c.execute('SELECT * FROM my_table WHERE entry_state="Exited"')
        data = self.c.fetchall()
        return data

    def read_last_entry(self):
        self.c.execute('SELECT * FROM my_table ORDER BY id DESC LIMIT 1')
        return self.c.fetchone()[0]

    def write_to_file(self,data, filename):
        self.data = data
        self.filename = filename
        with open(self.filename, 'wb') as f:
            f.write(data)

    def change_state(c_id,time):
        self.c.execute('SELECT * FROM my_table')
        self.c.execute('UPDATE my_table SET entry_state="Exited", exit_time=(?) WHERE id=(?)',(time,c_id))
        self.conn.commit()

    def sync_database(self,temp_database,entered_ids):
        for i,t_d in enumerate(temp_database):
            a,b,c,d,e = t_d
            if a not in entered_ids:
                self.data_entry(a,b,c,d,e)
                entered_ids.append(a)
            if c=='Exited':
                temp_database.remove(temp_database[i])


