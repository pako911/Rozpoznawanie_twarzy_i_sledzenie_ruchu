import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import urllib.request


def face_recognition(self):
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture(cv2.CAP_DSHOW)
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainingData.yml")
    _id = 0
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    qm = QtWidgets.QMessageBox
    ret = qm.question(self, ' ', "Do you want to record?", qm.Yes | qm.No)
    if ret == qm.Yes:
        record = True
        self.qbox = QtWidgets.QLineEdit()
        video_name, ok = QtWidgets.QInputDialog.getText(self, ' ', 'Enter video name', QtWidgets.QLineEdit.Normal,
                                                        'camera_video')
        if ok:
            self.qbox.setText(str(video_name))
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            video = cv2.VideoWriter(video_name + '.avi', fourcc, 20.0, (640, 480))
        else:
            record = False
    else:
        record = False
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.05, 5)
        blabla = np.array(faces)
        i = blabla.size
        i = i / 4
        cv2.putText(img, "number of people: " + str(i), (10, 50), font_face, 0.7, (255, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _id, conf = rec.predict(gray[y:y + h, x:x + w])
            profile = self.get_profile(_id)

            if profile is not None:
                if conf > 60:  # prawdopodobieństwo poprawnego wykrycia twarzy (im niższa liczba tym jest ono większe)
                    cv2.putText(img, "unknown", (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                else:
                    _id, conf = rec.predict(gray[y:y + h, x:x + w])
                    profile = self.get_profile(_id)
                    cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                    cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                    cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)
        cv2.putText(img, "Press [ESC] to exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("Main camera view", img)
        if record:
            video.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


def ip_camera_face_recognition(self):
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    ip_addr = self.ui.ipCameraAddress.toPlainText()
    url = 'http://' + ip_addr + ':8080/shot.jpg'
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer\\trainingData.yml")
    _id = 0
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    qm = QtWidgets.QMessageBox
    ret = qm.question(self, ' ', "Do you want to record?", qm.Yes | qm.No)
    if ret == qm.Yes:
        record = True
        self.qbox = QtWidgets.QLineEdit()
        video_name, ok = QtWidgets.QInputDialog.getText(self, '', 'Enter video name')
        if ok:
            self.qbox.setText(str(video_name))
        else:
            video_name = "camera_video"
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(video_name+'.avi', fourcc, 30.0, (640, 480))
    else:
        record = False
    while True:
        try:
            img_resp = urllib.request.urlopen(url)
        except urllib.error.URLError:
            errmsg = QtWidgets.QErrorMessage(self)
            errmsg.setWindowTitle("Error")
            errmsg.setWindowIcon(QtGui.QIcon('error-flat.png'))
            errmsg.showMessage('Invalid Ip address')
            break

        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.05, 5)
        blabla = np.array(faces)
        i = blabla.size
        i = i / 4
        cv2.putText(img, "number of people: " + str(i), (10, 50), font_face, 1, (255, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            _id, conf = rec.predict(gray[y:y + h, x:x + w])
            profile = self.get_profile(_id)
            if profile is not None:
                cv2.putText(img, str(profile[1]), (x, y + h + 30), font_face, 1, (255, 0, 0), 2)
                # zamist str(id) -> profile
                cv2.putText(img, str(profile[2]), (x, y + h + 60), font_face, 1, (255, 0, 0), 2)
                cv2.putText(img, str(profile[3]), (x, y + h + 90), font_face, 1, (255, 0, 0), 2)
        cv2.putText(img, "Press [ESC] to exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow("IP camera view", img)
        if record:
            video.write(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

