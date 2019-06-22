import cv2
import numpy as np
import os
import sqlite3
import sys
import urllib.request

from PIL import Image
from PyQt5 import QtWidgets, QtCore, QtGui
from AppGUI import Ui_MainWindow


class My_Form(QtWidgets.QMainWindow):
    def __init__(self):
        super(My_Form, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon('web-camera.png'))
        self.ui.pushButton.clicked.connect(self.face_recognition)
        self.ui.pushButton_2.clicked.connect(self.display_DB)
        self.ui.ipCameraButton.clicked.connect(self.ip_camera_face_recognition)
        self.ui.pushButton_6.clicked.connect(self.motion_detector_cam)
        self.ui.pushButton_3.clicked.connect(self.db_edit)
        self.ui.pushButton_4.clicked.connect(self.create_data_set)
        self.ui.pushButton_5.clicked.connect(self.create_training_file)

    def create_data_set(self):
        face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cam = cv2.VideoCapture(0)
        self.qbox = QtWidgets.QLineEdit()
        id, ok = QtWidgets.QInputDialog.getInt(self, ' ', 'Enter your id')
        if ok:
            name, ok = QtWidgets.QInputDialog.getText(self, '', 'Enter your name')
            if ok:
                age, ok = QtWidgets.QInputDialog.getInt(self, '', 'Enter your age')
                if ok:
                    gender, ok = QtWidgets.QInputDialog.getItem(self, 'Select your gender', 'gender',
                                                                ('Male', 'Female'), 0, False)
                    if ok:
                        self.insert_or_update(id, "\"" + name + "\"", age, "\"" + gender + "\"")
                        sample_num = 0

                        while True:
                            ret, img = cam.read()
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            faces = face_detect.detectMultiScale(gray, 1.3, 2)
                            for (x, y, w, h) in faces:
                                sample_num = sample_num + 1
                                cv2.imwrite("dataSet/User." + str(id) + "." + str(sample_num) + ".jpg",
                                            gray[y:y + h, x:x + w])
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

    def get_images_with_id(self, file_name):
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

    def create_training_file(self):
        ids, faces = self.get_images_with_id(self.path)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save('recognizer/trainingData.yml')
        cv2.destroyAllWindows()

    def get_profile(self, _id):
        conn = sqlite3.connect("FaceBaseGit.db")
        cmd = "SELECT * FROM People WHERE ID=" + str(_id)
        cursor = conn.execute(cmd)
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile

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
            faces = face_detect.detectMultiScale(gray, 1.5, 5)
            blabla = np.array(faces)
            i = blabla.size
            i = i / 4
            cv2.putText(img, "number of people: " + str(i), (10, 50), font_face, 0.7, (255, 255, 0), 2)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                _id, conf = rec.predict(gray[y:y + h, x:x + w])
                profile = self.get_profile(_id)

                if profile is not None:
                    if conf > 100:  # prawdopodobieństwo poprawnego wykrycia twarzy (im niższa liczba tym jest ono większe)
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
            faces = face_detect.detectMultiScale(gray, 1.3, 2)
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

    def motion_detector_cam(self):
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
        mog2 = cv2.createBackgroundSubtractorMOG2()

        while cap.isOpened():
            ret, frame = cap.read()
            frame1 = frame
            fgmask = mog2.apply(frame1)
            cv2.putText(fgmask, "Press [ESC] to exit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('Motion detection', fgmask)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def display_DB(self):
        conn = sqlite3.connect("FaceBaseGit.db")
        query = "SELECT * FROM People"
        result = conn.execute(query)

        self.ui.tableWidget.setRowCount(0)
        for row_number, row_data in enumerate(result):
            self.ui.tableWidget.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.ui.tableWidget.setItem(row_number, column_number,
                                            QtWidgets.QTableWidgetItem(str(data)))
        conn.close()

    def insert_or_update(self, _id, name, age, gender):  # funkcja do modyfikowania bazy danych
        conn = sqlite3.connect("FaceBaseGit.db")
        cmd = "SELECT * FROM People WHERE ID =" + str(_id)
        cursor = conn.execute(cmd)
        does_record_exists = 0

        for _ in cursor:
            does_record_exists = 1
        if does_record_exists == 1:
            cmd = "UPDATE People SET Age=" + str(age) + "," + "Gender=" + str(gender) + "," + "Name=" + str(
                name) + " WHERE ID =" + str(_id)
        else:
            cmd = "INSERT INTO People(ID,Name,Age,Gender) Values(" + str(_id) + "," + str(name) + "," + str(
                age) + "," + str(gender) + ")"
        conn.execute(cmd)
        conn.commit()
        conn.close()

    def db_edit(self):  # funkcja do modyfikowania bazy danych
        self.qbox = QtWidgets.QLineEdit()
        id, ok = QtWidgets.QInputDialog.getInt(self, ' ', 'Enter your id')
        if ok:
            name, ok = QtWidgets.QInputDialog.getText(self, '', 'Enter your name')
            if ok:
                age, ok = QtWidgets.QInputDialog.getInt(self, '', 'Enter your age')
                if ok:
                    gender, ok = QtWidgets.QInputDialog.getItem(self, 'Select your gender', 'gender',
                                                                ('Male', 'Female'), 0, False)
                    if ok:
                        self.insert_or_update(id, "\"" + name + "\"", age, "\"" + gender + "\"")


def main():
    app = QtWidgets.QApplication(sys.argv)
    my_app = My_Form()
    my_app.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
