import cv2 
import sys 
import os 
import numpy as np 
import imutils
from os import listdir
import mediapipe as mp 

def parsefiles(dataDir:str,allLabels:list):
    """
    List all files of one label 
    load and append arrays to one big array, then save 
    """
    bigArr = None
    d = {} 
    files = [f for f in listdir(dataDir) if f.endswith('.npy') and "full" not in f]
    print(files)
    # TODO: Move this check to the loop below 
    if len(files) <2:
        print("Not enough files in folder to parse ")
        return None

    labeled= [[f for f in listdir(dataDir) if f.endswith('.npy') and "full" not in f and label in f] for label in allLabels] 


    d = dict(zip(allLabels,labeled))
    
    for label, files in d.items(): 
        for f in files:
            if bigArr is None: 
                bigArr = np.load(os.path.join(dataDir,f),allow_pickle=True)
                os.remove(os.path.join(dataDir,f))
        
            else:
                data = np.load(os.path.join(dataDir,f),allow_pickle=True)
                bigArr = np.vstack((bigArr,data))
                os.remove(os.path.join(dataDir,f))
                
        N = bigArr.shape[0]
        delim = "-"
        name = delim.join([label.lower(),"full",str(N)])    
        np.save(os.path.join(dataDir,name),bigArr)
        bigArr = None

    return None

def getName(dataDir:str,label:str,N:int):
    delim = "-"
    highest = 1
    LabelFiles = [f for f in listdir(dataDir) if label.lower() in f and f.endswith('.npy')]
    
    for f in LabelFiles:
            index = f.split(delim,2)[-1]
            
            index = int(index.split('.',1)[0])
            if index == highest:
                highest +=1 
    
    name = delim.join([label.lower(),str(N),str(highest)])
    return name


def getInput():
    label = input("Please enter the label: ")

    assert label.isalpha(), "Label must be a string"
    maxN = None
    maxInput= input("Please enter maximum number of samples: ")

    if maxInput:
        maxN = int(maxInput)
    else:
        maxN = 50
    
    return maxN,label

def getSamples(label,maxN,show =False):
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    train_data = []# [label,rawImage,poseImage,mpresults]
    POSE_LANDMARKS_INDEX = [x for x in range(12,23)]
    i = 0 
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        while (cap.isOpened()) and i < maxN:
            succes,image = cap.read()
            if not succes:
                print("Couldnt load webcam")
                break 
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            Posekeypoints = []
            RightHkeypoints = [] 
            LeftHkeypoints = []
            if results.pose_landmarks is not None:
                for idx,data_point in enumerate(results.pose_landmarks.landmark):
                    if idx in POSE_LANDMARKS_INDEX:
                        
                        Posekeypoints.append({
                                    'X': data_point.x,
                                    'Y': data_point.y,
                                    'Z': data_point.z,
                                    'Visibility': data_point.visibility,
                                    }) 

            
            
            if results.right_hand_landmarks is not None:
                for idx,data_point in enumerate(results.right_hand_landmarks.landmark):       
                    RightHkeypoints.append({
                                'X': data_point.x,
                                'Y': data_point.y,
                                'Z': data_point.z,
                                'Visibility': data_point.visibility,
                                }) 
            if  results.left_hand_landmarks is not None:
                for idx,data_point in enumerate(results.left_hand_landmarks.landmark):       
                    LeftHkeypoints.append({
                                'X': data_point.x,
                                'Y': data_point.y,
                                'Z': data_point.z,
                                'Visibility': data_point.visibility,
                                })         

            
            train_data.append([label,Posekeypoints,LeftHkeypoints,RightHkeypoints])
            
            if show:
                mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                cv2.imshow('MediaPipe Holistic', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    
                    break
            i +=1 

    cap.release()           
    cv2.destroyAllWindows()


    
    return np.array(train_data,dtype=object)




if __name__ == '__main__':
    allLabels = []
    cwd = os.getcwd() #NOTE: this gives the parent directory from where the file is called 
    parentDir = os.path.dirname(os.path.dirname(__file__)) #NOTE: this gives the parent directory of the file 
    
    dataDir = os.path.join(parentDir,"data")
    data =  getSamples('test',100,True)
    name = getName(dataDir,'test',100)

    np.save(os.path.join(dataDir,name + ".npy"),data)
    # try:

    #     while (True):
    #         maxN,label = getInput()
    #         
    #         data = getSamples(label,maxN)
            
    #        
    #         allLabels.append(label)
    # except KeyboardInterrupt:
    #     pass        
        
    
    print('exited while loop, parsing data folder')
