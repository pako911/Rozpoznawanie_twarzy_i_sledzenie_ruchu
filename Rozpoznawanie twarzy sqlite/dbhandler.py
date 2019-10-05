import sqlite3
import os
from PyQt5 import QtWidgets
from AppGUI import Ui_MainWindow


class DBHandler:
    def __init__(self):
        super(DBHandler, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

    def display_db(self):
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

    def update_db(self, _id, name, age, gender):  # funkcja do modyfikowania bazy danych
        conn = sqlite3.connect("FaceBaseGit.db")
        cmd = "SELECT * FROM People WHERE ID =" + str(_id)
        cursor = conn.execute(cmd)
        does_record_exists = 0

        for _ in cursor:
            does_record_exists = 1
        if does_record_exists == 1:
            cmd = "UPDATE People SET Age=" + str(age) + "," + "Gender=" + str(gender) + "," + "Name=" + str(
                name) + " WHERE ID =" + str(_id)

        conn.execute(cmd)
        conn.commit()
        conn.close()

    def insert_person(self, name, age, gender):
        conn = sqlite3.connect("FaceBaseGit.db")
        cursor = conn.execute("SELECT MAX(id) FROM People")
        max_id = cursor.fetchone()[0]
        if max_id != None:
            id = max_id + 1
        else:
            id = 1
        cmd = "INSERT INTO People(ID,Name,Age,Gender) Values(" + str(id) + "," + str(name) + "," + str(
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
                        self.db_update(id, "\"" + name + "\"", age, "\"" + gender + "\"")

    def delete_user(self):
        self.qbox = QtWidgets.QLineEdit()
        _id, ok = QtWidgets.QInputDialog.getInt(self, ' ', 'Enter your id')
        conn = sqlite3.connect("FaceBaseGit.db")
        cmd = "SELECT * FROM People WHERE ID =" + str(_id)
        cursor = conn.execute(cmd)
        does_record_exists = 0

        for _ in cursor:
            does_record_exists = 1
        if does_record_exists == 1:
            cmd = "DELETE FROM People where ID = " + str(_id)
            my_dir = "C:/Users/pako9/Documents/GitHub/Rozpoznawanie_twarzy_i_sledzenie_ruchu/" \
                     "Rozpoznawanie twarzy sqlite/dataSet/"
            for fname in os.listdir(my_dir):
                if fname.startswith("User." + str(_id)):
                    os.remove(os.path.join(my_dir, fname))

        conn.execute(cmd)
        conn.commit()
        conn.close()
