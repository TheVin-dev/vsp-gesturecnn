import cv2 
import sys 
import os 
import numpy as np 
import imutils
from os import listdir

def parsefullfiles(dataDir:str,labels:list):
    """
    List all  FULL files of one label 
    load and append arrays to one big array, then save 
    """


    return None

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
                #os.remove(os.path.join(dataDir,f))
        
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

def getSamples(label):
    cap = cv2.VideoCapture(0)
    imgs = []
    train_data = []# [label,imaege]

    while (cap.isOpened) and len(imgs) < maxN: 
        succes,image = cap.read() 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if (not succes):
            print("Couldnt read frame! \n Exiting...")
            sys.exit(0)
            break

        downscaled = imutils.resize(image,width= 300)
        cv2.imshow('Downsized',downscaled)
        imgs.append(downscaled)
        train_data.append([label,downscaled])
        if cv2.waitKey(2) == 27: 
            break

            
    cv2.destroyAllWindows()
    return np.array(train_data,dtype=object)




if __name__ == '__main__':
    allLabels = []
    cwd = os.getcwd() #NOTE: this gives the parent directory from where the file is called 
    parentDir = os.path.dirname(os.path.dirname(__file__)) #NOTE: this gives the parent directory of the file 
    print(parentDir)
    dataDir = os.path.join(parentDir,"data")
    try:
        while (True):
            maxN,label = getInput()
            name = getName(dataDir,label,maxN)
            data = getSamples(label)
            np.save(os.path.join(dataDir,name + ".npy"),data)
            allLabels.append(label)
    except KeyboardInterrupt:
        pass        
        

    parsefiles(dataDir,allLabels)