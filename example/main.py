import sys 
from PyQt5 import uic 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 
import cv2 
import mediapipe as mp 


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        uic.loadUi('mainwindow.ui',self)
        self._timer =QTimer(self)
        self.camera = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic # https://google.github.io/mediapipe/solutions/holistic.html 
        # callbacks
        self.stopCam.clicked.connect(self.stopTimer)
        self.startCam.clicked.connect(self.startTimer)
        self.TrainModel.clicked.connect(self.OnTrain)
        self.generateData.clicked.connect(self.OnGenerateData)
        self.previewGenerated.clicked.connect(self.OnPreview)

            
        self._timer.timeout.connect(self.on_timeout)
        self._timer.setInterval(5)

        
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

    def OnTrain(self):
        msg = QMessageBox()
        msg.setText("Training not yet implemented! :( ")
        msg.exec()

    def OnGenerateData(self):
        msg = QMessageBox()
        msg.setText("Generating not yet implemented! :( ")
        msg.exec()

    def OnPreview(self):
        """
        Show samples of generated data 
        """
        pass

    
    def OnTestModel(self):
       
        self._timer.stop()

        with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic: 
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Dit moet naar de Qlabel als Pixmap
            # mp_drawing.draw_landmarks(
            #         image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(
            #         image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # mp_drawing.draw_landmarks(
            #         image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            # cv2.imshow('MediaPipe Holistic', image)


if __name__ == "__main__":

    app = QApplication(sys.argv)
   
    mainWindow = MainWindow()
    mainWindow.show()
    mainWindow.activateWindow()
    sys.exit(app.exec())
