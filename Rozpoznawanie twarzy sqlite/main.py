import cv2
import numpy as np
import os
from PIL import Image
import sqlite3
import urllib.request


def face_detection():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_detect = cv2.CascadeClassifier('haarcascade_smile.xml')
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 2)
        eyes = eye_detect.detectMultiScale(gray, 1.3, 5)
        smile = smile_detect.detectMultiScale(gray, 1.3, 5)
        i = 0
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            i = i + 1
        #for (x, y, w, h) in eyes:
         #   cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #for (x, y, w, h) in smile:
         #   cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Face", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def insert_or_update(_id, name):
    conn = sqlite3.connect("FaceBaseGit.db")
    cmd = "SELECT * FROM People WHERE ID =" + str(_id)
    cursor = conn.execute(cmd)
    does_record_exists = 0
    for _ in cursor:
        does_record_exists = 1
    if does_record_exists == 1:
        cmd = "UPDATE People SET Name=" + str(name) + " WHERE ID =" + str(_id)
    else:
        cmd = "INSERT INTO People(ID,Name) Values(" + str(_id) + "," + str(name) + ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()



def create_data_set():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    id = input('enter user id')
    name = input('enter your name')  # ma wartość not null w bazie danych
    insert_or_update(id, name)
    sample_num = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            sample_num = sample_num + 1
            cv2.imwrite("dataSet/User." + str(id) + "." + str(sample_num) + ".jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if sample_num > 20:
            break
    cam.release()
    cv2.destroyAllWindows()


recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'


def get_images_with_id(file_name):
    image_paths = [os.path.join(file_name, f) for f in os.listdir(file_name)]
    faces = []
    ids = []
    for imagePath in image_paths:
        face_img = Image.open(imagePath).convert('L')
        face_np = np.array(face_img, 'uint8')
        _id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(face_np)
        print(_id)
        ids.append(_id)
        cv2.imshow("training", face_np)
        cv2.waitKey(10)
    return ids, faces


def create_training_file():
    ids, faces = get_images_with_id(path)
    recognizer.train(faces, np.array(ids))
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()


def get_profile(_id):
    conn = sqlite3.connect("FaceBaseGit.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(_id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


def face_recognition():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainingData.yml")
    _id = 0
    font_face = cv2.FONT_HERSHEY_SIMPLEX
   # model = cv2.eigen
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.5, 5)
        blabla = np.array(faces)
        i = blabla.size
        i = i / 4
        cv2.putText(img, str(i),(100,200), font_face, 1, (255,255,0),2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _id, conf = rec.predict(gray[y:y + h, x:x + w])
            profile = get_profile(_id)
            print(conf)
            '''if profile is not None:
                cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)'''
            if profile is not None:
                if conf > 100:  # prawdopodobieństwo poprawnego wykrycia twarzy (im niższa liczba tym jest ono większe)
                    cv2.putText(img, "unknown", (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                else:
                    _id, conf = rec.predict(gray[y:y + h, x:x + w])
                    profile = get_profile(_id)
                    cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                    cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                    cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)

        cv2.imshow("Face", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


def ip_camera():
    url = 'http://10.5.5.26:8080/shot.jpg'  # trzeba bedzie zmienic
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            break


def ip_camera_face_detection():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    url = 'http://10.5.5.26:8080/shot.jpg'  # trzeba bedzie zmienic
    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            break


def ip_camera_face_recognition():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    url = 'http://10.5.5.26:8080/shot.jpg'  # trzeba bedzie zmienic
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainingData.yml")
    _id = 0
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _id, conf = rec.predict(gray[y:y + h, x:x + w])
            profile = get_profile(_id)
            if profile is not None:
                cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                # zamist str(id) -> profile
                cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            break


def detecting_object():
    lowerBound = np.array([110,50,50])
    upperBound = np.array([130,255,255])

    cam = cv2.VideoCapture(0)
    kernelOpen = np.ones((5,5))
    kernelClose = np.ones((20,20))

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret,img = cam.read()
        img = cv2.resize(img,(340,220))

        #convert BGR to HSV
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #create the Mask
        mask = cv2.inRange(imgHSV, lowerBound,upperBound)

        #morphology
        maskOpen = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        maskFinal = maskClose

        conts,h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(img,conts,-1,(255,0,0),3)

        for i in range(len(conts)):
            x,y,w,h = cv2.boundingRect(conts[i])
            cv2.rectangle(img, (x,y), (x+w,y+h),(0,0,255),2)
            cv2.putText(img,str(i+1),(x,y+h),font,1,(0,255,255))
        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("mask", mask)
        cv2.imshow("cam", img)
        if cv2.waitKey(1) == ord('q'):
            break


def person_detection():
    #cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    cap = cv2.VideoCapture('walking_people.mp4')
    human_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        human = human_cascade.detectMultiScale(gray,1.1,4)

        for(x,y,w,h) in human:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,220),4)
        cv2.imshow('video', frame)
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()



def motion_detector():
    #cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # nagranie z kamery
    cap = cv2.VideoCapture('walking_people.mp4')  # nagranie ludzi z yt
    mog2 = cv2.createBackgroundSubtractorMOG2()

    # zapisywanie wideo
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # original_out = cv2.VideoWriter('original_out.avi', fourcc, 20.0, (640, 480))
    # mog2_out = cv2.VideoWriter('mog2_out.avi', fourcc, 20.0, (640, 480), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        frame1=frame/8
        fgmask = mog2.apply(frame1)
        cv2.imshow('frame1', frame)
        cv2.imshow('frame2', fgmask)
        # original_out.write(frame)  # zapisywanie obrazu orginalnego
        # mog2_out.write(fgmask) # zapisywanie wykrywania ruchu
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    # original_out.release()
    # mog2_out.release()
    cv2.destroyAllWindows()

def motion_detector_cam():
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)  # nagranie z kamery
    #cap = cv2.VideoCapture('walking_people.mp4')  # nagranie ludzi z yt
    mog2 = cv2.createBackgroundSubtractorMOG2()

    # zapisywanie wideo
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # original_out = cv2.VideoWriter('original_out.avi', fourcc, 20.0, (640, 480))
    # mog2_out = cv2.VideoWriter('mog2_out.avi', fourcc, 20.0, (640, 480), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        frame1=frame
        fgmask = mog2.apply(frame1)
        cv2.imshow('frame1', frame)
        cv2.imshow('frame2', fgmask)
        # original_out.write(frame)  # zapisywanie obrazu orginalnego
        # mog2_out.write(fgmask) # zapisywanie wykrywania ruchu
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    # original_out.release()
    # mog2_out.release()
    cv2.destroyAllWindows()

def display_DB():
    conn = sqlite3.connect("FaceBaseGit.db")
    cmd = "SELECT * FROM People"
    cursor = conn.execute(cmd)

    for row in cursor:
        print(row)

def count_DB():
    conn = sqlite3.connect("FaceBaseGit.db")
    cmd = "SELECT COUNT(ID) FROM People"
    cursor = conn.execute(cmd)
    i =0
    for row in cursor:
        i = int(row[0])
    print(i)


def insert_or_update2(_id,name, age,gender):# funkcja do modyfikowania bazy danych
    conn = sqlite3.connect("FaceBaseGit.db")
    cmd = "SELECT * FROM People WHERE ID =" + str(_id)
    cursor = conn.execute(cmd)
    does_record_exists = 0
    for _ in cursor:
        does_record_exists = 1
    if does_record_exists == 1:
        cmd = "UPDATE People SET Age=" + str(age) + "," + "Gender=" + str(gender)+"," + "Name=" +str(name) + " WHERE ID =" + str(_id)
    else:
        cmd = "INSERT INTO People(ID,Name,Age,Gender) Values(" + str(_id) + "," + str(name) +"," + str(age) + "," + str(gender) + ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()


def create_data_set2(): # funkcja do modyfikowania bazy danych
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    id = input('enter user id')
    name = input('enter your name')  # ma wartość not null w bazie danych
    age = input('enter your age')
    gender = input('enter your gender')
    insert_or_update2(id, name,age,gender)



if __name__ == "__main__":
    #face_detection()
    # create_data_set()
    # get_images_with_id(file_name)
    # create_training_file()
    # ip_camera()
    # ip_camera_face_detection()
    # ip_camera_face_recognition()
    #insert_or_update2(7,names, 23, 1)
   # display_DB()
   # create_data_set2()
   # display_DB()
   # count_DB()
    #person_detection()
    #motion_detector
    #motion_detector_cam()
    face_recognition()





