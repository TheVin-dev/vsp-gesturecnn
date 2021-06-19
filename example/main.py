import sys 
from PyQt5 import uic 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 
import cv2 



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        uic.loadUi('mainwindow.ui',self)
        self._timer =QTimer(self)
        self.camera = cv2.VideoCapture(0)
        self.stopCam.clicked.connect(self.stopTimer)
        self.startCam.clicked.connect(self.startTimer)
        
        self.display_height = 300 
        self.display_width = 300

            
        self._timer.timeout.connect(self.on_timeout)
        self._timer.setInterval(10)

        
    def on_timeout(self):
        
        
        ok, image = self.camera.read() 
        if ok:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format =QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.image_frame.size())
            p = QPixmap(p)
           
            self.image_frame.setPixmap(p)
            
            
            return None
    def stopTimer(self):
        self._timer.stop()
        # TODO: Set default image
    def startTimer(self):
        self._timer.start()





if __name__ == "__main__":

    app = QApplication(sys.argv)
   
    mainWindow = MainWindow()
    mainWindow.show()
    mainWindow.activateWindow()
    sys.exit(app.exec())
