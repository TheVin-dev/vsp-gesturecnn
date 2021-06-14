import cv2 
import sys 
import os 
import numpy as np 
import random 
import imutils
from os import listdir

# [[j for j in range(5)] for i in range(5)]
# testimg = [[random.randint(0,1920) for x in range(1080)] for y in range(1920)]
# testimg = np.asarray(testimg,dtype=np.float32)
# imglist = [testimg for x in range(10)]
# imgs = imglist


train_data = []# [label,imaege]

cwd = os.getcwd() 
parentDir = os.path.dirname(cwd)
dataDir = os.path.join(parentDir,"data")

# data folder structures: 
    # All data in one folder as npz [label, image]
cap = cv2.VideoCapture(0)
labels = None 
inputLabels = input("Please enter the labels: ")
labels = inputLabels
maxN = None
maxInput= input("Please enter maximum number of samples: ")




if maxInput:
    maxN = int(maxInput)
else:
    maxN = 1000


imgs = []
while (cap.isOpened) and len(imgs) < maxN: 
    succes,image = cap.read() 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if (not succes):
        print("Couldnt read frame! \n Exiting...")
        sys.exit(0)
        break

    #cv2.imshow('webcam',image)
    downscaled = imutils.resize(image,width= 300)
    cv2.imshow('Downsized',downscaled)
    if len(imgs) >= maxN:
        print("Done with collecting")

        break
    
    imgs.append(downscaled)
    train_data.append([labels,downscaled])
    if cv2.waitKey(2) == 27: 
        break

cv2.destroyAllWindows()

delim = "-"

onlyfiles = [f for f in listdir(dataDir) if f.endswith(".npy")]
highest = 1 
for f in listdir(dataDir):
    if f.endswith("*.npy"):
        
        index = f.split(delim,2)[-1]
        print(index)
        if index > highest:
            highest = index
print(highest)
name = delim.join([labels.lower(),"d",str(highest)])
train_data = np.array(train_data)
np.save(os.path.join(dataDir,name + ".npy"),train_data)



