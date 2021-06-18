import sys 
from PySide6 import QtCore, QtWidgets, QtGui
from test import Ui_MainWindow  

class MyWindow(QtWidgets.QMainWindow,Ui_MainWindow ):
    
    def __init__(self):
        super(MyWindow,self).__init__()
        self.setupUi(self)
        # self.b1.clicked.connect(self.increase_counter)
        # self.counter = 0 

    @QtCore.Slot()
    def increase_counter(self):
        self.counter +=1 
        self.label.setText(str(self.counter))


 
app = QtWidgets.QApplication(sys.argv)
mainWindow = MyWindow()
mainWindow.show()
sys.exit(app.exec_())