import cv2
import numpy as np
import os
from PIL import Image
import sqlite3

import urllib.request


def wykrywanieTwarzy():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyeDetect = cv2.CascadeClassifier('haarcascade_eye.xml')
    smileDetect = cv2.CascadeClassifier('haarcascade_smile.xml')
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)

    while(True):
        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 2)
        eyes = eyeDetect.detectMultiScale(gray, 1.3,5)
        smile = smileDetect.detectMultiScale(gray, 3,5)
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for(x, y, w, h) in eyes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for(x, y, w, h) in smile:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow("Face", img)
        if(cv2.waitKey(1)==ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()

def insertOrUpdate(Id,Name):
    conn = sqlite3.connect("Facebase.db")
    cmd="SELECT * FROM People WHERE ID =" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name="+str(Name) + " WHERE ID =" + str(Id)
    else:
        cmd = "INSERT INTO People(ID,Name) Values(" + str(Id) + "," + str(Name) + ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

def tworzenieDataSet():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(0)

    id = input('enter user id')
    name = input('enter your name') # ma wartość not null w bazie danych
    insertOrUpdate(id,name)
    sampleNum = 0

    while(True):
        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 2)
        for(x, y, w, h) in faces:
            sampleNum=sampleNum+1
            cv2.imwrite("dataSet/User." + str(id)+"." + str(sampleNum)+ ".jpg", gray[y:y+h,x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(100)
        cv2.imshow("Face", img)
        cv2.waitKey(1)
        if(sampleNum>20):
            break
    cam.release()
    cv2.destroyAllWindows()

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'
def getImagesWithID(path):


    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs=[]
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs,faces

def tworzeniePlikuTreningowego():
    Ids, faces = getImagesWithID(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()


def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID=" + str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile



def rozpoznawanieTwarz():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    rec =cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainingData.yml")
    id=0
    fontface=cv2.FONT_HERSHEY_SIMPLEX
    while(True):
        ret,img = cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 2)
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id,conf = rec.predict(gray[y:y+h,x:x+w])
            profile = getProfile(id)
            if(profile!=None):
                cv2.putText(img, str(profile[1]), (x, y + h + 30), fontface, 1, (255, 0, 0), 2) #zamista str(id) -> profile
                cv2.putText(img, str(profile[2]), (x, y + h + 60), fontface, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), fontface, 1, (255, 0, 0), 2)

            '''if(id==1):
                id="Remeq"
            elif(id==2):
                id="Pan za STOCKA"
            elif(id==3):
                id="Pawel"'''

        cv2.imshow("Face", img)
        if(cv2.waitKey(1)==ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()

def KamerkaIP():
   url = 'http://10.5.5.54:8080/shot.jpg' # trzeba bedzie zmienic
   while(True):
       imgResp = urllib.request.urlopen(url)
       imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
       img=cv2.imdecode(imgNp,-1)
       cv2.imshow('test',img)
       if (cv2.waitKey(1) == ord('q')):
           break


def KamerkaIPwykrywanieTwarzy():
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    url = 'http://10.5.5.54:8080/shot.jpg'  # trzeba bedzie zmienic
    while (True):
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 2)
        for(x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('test', img)
        if (cv2.waitKey(1) == ord('q')):
            break

if __name__== "__main__":
    #wykrywanieTwarzy()
    #tworzenieDataSet()
    #getImagesWithID(path)
    #tworzeniePlikuTreningowego()
    #rozpoznawanieTwarz()
    #KamerkaIP()
    KamerkaIPwykrywanieTwarzy()