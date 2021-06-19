import numpy as np 
import os 
import numpy as np 
import sys
import cv2 
import time 


def loadData(labels,N=100):

    cwd = os.getcwd() 
    parentDir = os.path.dirname(cwd)
    dataDir = os.path.join(parentDir,"data")
    data = None # numpy array
    files = [[f for f in os.listdir(dataDir) if label in f and  f.endswith('.npy') and  "full" in f] for label in labels]
    
    if len(files) == 0:
        sys.exit("No Files found")
    
    for f in files:
        
        n = f.split("-",3)[2].split(".")[0]
        print(f"found {n} samples in {f}")

        if data is None:
            tmpdata = np.load(os.path.join(dataDir,f),allow_pickle=True)
            length = tmpdata.shape[0]
            if N < length: 
                data = tmpdata[:N]
            else:
                data = tmpdata
        else:
            d = np.load(os.path.join(dataDir,f),allow_pickle=True)
            length = data.shape[0]
            if N < length:
                np.append(data,d[:N])
                return data
            if data.shape[0] < N:
                np.append(data,d)
            
    return data 

def CheckImages(data):

    """
    Helper function to look at data 
    """
    print("Press enter to go to the next image")
    i=1
    while i < rawdata.shape[0]:
        
        data= rawdata[i][:]

       
        label, img = data
        print(f"showing {i} image")
        cv2.imshow('Loaded image',img)

        p =  cv2.waitKey(500)
        while p != 32:
            p =  cv2.waitKey(500)

            time.sleep(0.1)

        i+=1

    return None


if __name__ == "__main__":
    
    rawdata = loadData('stop',200)
    CheckImages(rawdata)
  