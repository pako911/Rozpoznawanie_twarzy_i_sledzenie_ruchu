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
        smile = smile_detect.detectMultiScale(gray, 3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h) in eyes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        for (x, y, w, h) in smile:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("Face", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def insert_or_update(_id, name):
    conn = sqlite3.connect("Facebase.db")
    cmd = "SELECT * FROM People WHERE ID =" + str(_id)
    cursor = conn.execute(cmd)
    does_record_exists = 0
    for row in cursor:
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
    conn = sqlite3.connect("FaceBase.db")
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
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _id, conf = rec.predict(gray[y:y + h, x:x + w])
            profile = get_profile(_id)
            if profile is not None:
                cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0),
                            2)  # zamista str(id) -> profile
                cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)

            '''if(id==1):
                id="Remeq"
            elif(id==2):
                id="Pan za STOCKA"
            elif(id==3):
                id="Pawel"'''

        cv2.imshow("Face", img)
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


def ip_camera():
    url = 'http://10.5.5.54:8080/shot.jpg'  # trzeba bedzie zmienic
    while True:
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        cv2.imshow('test', img)
        if cv2.waitKey(1) == ord('q'):
            break


def ip_camera_face_detection():
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    url = 'http://10.5.5.54:8080/shot.jpg'  # trzeba bedzie zmienic
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
    url = 'http://10.5.5.54:8080/shot.jpg'  # trzeba bedzie zmienic
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


if __name__ == "__main__":
     face_detection()
    # create_data_set()
    # get_images_with_id(file_name)
    # create_training_file()
    # face_recognition()
    # ip_camera()
    # ip_camera_face_detection()
    #ip_camera_face_recognition()

    #taktyczny komentarz


