import sys 
from PyQt5 import uic 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import * 
import cv2 
import mediapipe as mp 
import numpy as np 
import os 
from os import listdir
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        uic.loadUi('mainwindow.ui',self)
        self._timer =QTimer(self)
        self.camera = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) # https://google.github.io/mediapipe/solutions/holistic.html 
        self.trainingData = []
        self.label = None
        # callbacks
        self.stopCam.clicked.connect(self.stopTimer)
        self.startCam.clicked.connect(self.startTimer)
        self.TrainModel.clicked.connect(self.OnTrain)
        self.generateData.clicked.connect(self.OnGenerateData)
        self.previewGenerated.clicked.connect(self.OnPreview)
        self.saveData.clicked.connect(self.onSave)
             
        self._timer.timeout.connect(self.on_timeout)
        self._timer.setInterval(5)

        self.TRAINING = False
        self.parentDir = os.path.dirname(os.path.dirname(__file__))
        self.dataDir = os.path.join(self.parentDir,"data")
        

    def on_timeout(self):

        if self.label is None:

            msg = QMessageBox()
            msg.setText("First set label")
            self._timer.stop()
            msg.exec()    
            return None

        ok, image = self.camera.read() 
        if not ok:
            return None
        
        if self.TRAINING:   

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results =  self.holistic.process(image)
            image.flags.writeable = True 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            self.trainingData.append([self.label,image,results.pose_landmarks,results.left_hand_landmarks,results.right_hand_landmarks])
            self.mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
            h, w, ch = image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format =QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.image_frame.size())
            p = QPixmap(p)
            self.image_frame.setPixmap(p) 
            
                
        else:
            self.drawImage(image)        

        return 
    def drawImage(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format =QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_frame.size())
        p = QPixmap(p)

        self.image_frame.setPixmap(p)   


    def stopTimer(self):
        self._timer.stop()
        # TODO: Set default image
        return None
    def startTimer(self):
        self._timer.start()

    def OnTrain(self):
        msg = QMessageBox()
        msg.setText("Training not yet implemented! :( ")
        msg.exec()
        return None

    def onSave(self):
        self._timer.stop()
        
        N, ok  = QInputDialog.getInt(self,"Number samples","Number of samples you want to keep? Leave empty to save all ")
        n = len(self.trainingData)
        data = np.array(self.trainingData,dtype=object)

        if (N <= 0): 

            name = self.getName(self.dataDir,self.label,n)
            print(name)
        else:
            name = self.getName(self.dataDir,self.label,N)
            data = data[:N]
            
            
       
        np.save(os.path.join(self.dataDir,name + ".npy"),data)
        print(f"Saved {data.shape[0]} samples: ")
        return None

    def OnGenerateData(self):
        self.TRAINING=True
        label, ok  = QInputDialog.getText(self,"Labeler","Label for the collected data")     
        if ok:
            self.label = label
        
        return None
    def OnPreview(self): # REMOVE THIS FUNCTION, NOT IMPORTANT
        """
        Show samples of generated data 
        """
        pass
        #np.random.choice(self.trainingData)
        # TODO: pop up 3 random frames with pose 
        # TODO: Give option to look at all data 
        

    def onReset(self):
        # TODO: Reset uit 
        # TODO: Blank image 
        # TODO: interrupt all processes? 
        pass
    
    def OnTestModel(self):
        #TODO: include case in on_timeout function 
        pass 

    def getName(self,dataDir:str,label:str,N:int):
        delim = "-"
        highest = 0
        LabelFiles = [f for f in listdir(dataDir) if label.lower() in f]

        for f in LabelFiles:
            if f.endswith(".npy"):

                index = f.split(delim,2)[-1]
                index = int(index.split('.',1)[0])
                if index == highest:
                    highest +=1 

        name = delim.join([label.lower(),str(N),str(highest)])
        return name


if __name__ == "__main__":

    app = QApplication(sys.argv)
   
    mainWindow = MainWindow()
    mainWindow.show()
    mainWindow.activateWindow()
    sys.exit(app.exec())
