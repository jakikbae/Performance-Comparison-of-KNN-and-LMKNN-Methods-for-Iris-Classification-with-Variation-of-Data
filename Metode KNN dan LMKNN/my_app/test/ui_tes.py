import sys
import os
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from models.knn_model import KNN
from models.lmknn_model import LMKNN
from models.lmknn_model import Proses_LMKNN

class TabelView(QtGui.QStandardItemModel):
    def __init__(self, data):
        super().__init__()
        for row in data:
            items = [QtGui.QStandardItem(str(cell)) for cell in row]
            self.appendRow(items)


class Ui_MainWindow(object):
    dataset = None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 602)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(30, 100, 311, 111))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(160, 60, 121, 31))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.tombolklasifikasi)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(160, 20, 121, 31))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setGeometry(QtCore.QRect(230, 25, 41, 21))
        font = QtGui.QFont()
        font.setFamily('Times New Roman')
        font.setPointSize(12)
        self.spinBox.setFont(font)
        self.spinBox.setObjectName("spinBox")
        self.spinBox.valueChanged.connect(self.get_value_spinbox)
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 30, 82, 17))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 60, 82, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 220, 771, 331))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        self.label.setGeometry(QtCore.QRect(20, 10, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(200, 70, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(570, 70, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.tableView_3 = QtWidgets.QTableView(self.groupBox_2)
        self.tableView_3.setGeometry(QtCore.QRect(10, 100, 501, 221))
        self.tableView_3.setObjectName("tableView_3")
        self.tableView = QtWidgets.QTableView(self.groupBox_2)
        self.tableView.setGeometry(QtCore.QRect(530, 100, 231, 221))
        self.tableView.setObjectName("tableView")
        self.tableView_2 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_2.setGeometry(QtCore.QRect(360, 51, 421, 161))
        self.tableView_2.setObjectName("tableView_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(40, 60, 111, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["Iris", "Irisd2"])
        self.comboBox.currentIndexChanged.connect(self.load_excel_data)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "KNN and LMKNN Classification"))
        self.groupBox.setTitle(_translate("MainWindow", "Parameter Klasifikasi"))
        self.pushButton_4.setText(_translate("MainWindow", "Klasifikasikan"))
        self.radioButton.setText(_translate("MainWindow", "KNN"))
        self.radioButton_2.setText(_translate("MainWindow", "LMKNN"))
        self.label.setText(_translate("MainWindow", "Hasil Klasifikasi"))
        self.label_2.setText(_translate("MainWindow", "Hasil Klasifikasi"))
        self.label_3.setText(_translate("MainWindow", "Confusion Matrix"))

    def get_value_spinbox(self):
        return self.spinBox.value()

    def tombolklasifikasi(self):
        k = self.get_value_spinbox()  # Ambil nilai k dari spinBox
        dataset_name = self.comboBox.currentText()
        if self.radioButton.isChecked():  # Jika RadioButton untuk KNN dipilih
            knn_model = KNN()  # Buat objek KNN
            scores, cm_df = knn_model.proses(k, dataset_name)
        elif self.radioButton_2.isChecked():  # Jika RadioButton untuk LMKNN dipilih
            lmknn_model = Proses_LMKNN()  # Buat objek LMKNN dengan nilai k yang diperoleh dari spinBox
            scores, cm_df = lmknn_model.proses(k, dataset_name)
        else:
            # Tidak ada RadioButton yang dipilih, tidak lakukan apa-apa
            return
            
        if scores is not None:
            self.display_results(scores, cm_df)


    def display_results(self, scores, cm_df):
        # Display scores
        model = TabelView(scores)
        self.tableView_3.setModel(model)
        
        # Display confusion matrix
        cm_model = TabelView(cm_df.values.tolist())
        self.tableView.setModel(cm_model)

    def load_excel_data(self):
        dataset_name = self.comboBox.currentText()
        data_dir = "C:/Users/jakikbae/OneDrive/Documents/BISMILLAH KERJA/Portofolio/Metode KNN dan LMKNN/my_app/data/"
        path = ""

        if dataset_name == 'Iris':
            path = os.path.join(data_dir, "Iris.xlsx")
        elif dataset_name == 'Irisd2':
            path = os.path.join(data_dir, "Irisd2.xlsx")
        else:
            return  # Exit if the dataset name is invalid

        if os.path.exists(path):
            df = pd.read_excel(path)
            self.display_excel_data(df)
        else:
            print(f"File {path} not found.")

    def display_excel_data(self, df):
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns)
        for row in df.itertuples(index=False):
            items = [QtGui.QStandardItem(str(cell)) for cell in row]
            model.appendRow(items)
        self.tableView_2.setModel(model)

        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
